import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
from collections import Counter

def create_sankey_diagram(df_transacoes, top_n=20):
    """Cria gráfico Sankey para visualizar fluxo de transações"""
    
    # Agrupar transações por origem e destino
    fluxo = df_transacoes.groupby(['ID_PGTO', 'ID_RCBE'])['VL'].sum().reset_index()
    
    # Pegar apenas os top N maiores fluxos
    fluxo_top = fluxo.nlargest(top_n, 'VL')
    
    # Criar lista de nós únicos
    nodes = list(set(fluxo_top['ID_PGTO'].tolist() + fluxo_top['ID_RCBE'].tolist()))
    node_dict = {node: i for i, node in enumerate(nodes)}
    
    # Preparar dados para Sankey
    source = [node_dict[row['ID_PGTO']] for _, row in fluxo_top.iterrows()]
    target = [node_dict[row['ID_RCBE']] for _, row in fluxo_top.iterrows()]
    value = fluxo_top['VL'].tolist()
    
    # Criar gráfico Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="lightblue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color="rgba(230, 0, 18, 0.3)"
        )
    )])
    
    fig.update_layout(
        title_text=f"Fluxo de Transações - Top {top_n} Maiores Fluxos",
        font_size=10,
        height=600
    )
    
    return fig

def create_network_graph(df_transacoes, cnpj_selecionado=None):
    """Cria gráfico de rede para visualizar relacionamentos"""
    
    # Se um CNPJ específico foi selecionado, filtrar apenas suas transações
    if cnpj_selecionado:
        df_transacoes = df_transacoes[
            (df_transacoes['ID_PGTO'] == cnpj_selecionado) | 
            (df_transacoes['ID_RCBE'] == cnpj_selecionado)
        ]
    
    # Criar grafo
    G = nx.Graph()
    
    # Calcular estatísticas para normalização dos pesos
    valores_transacoes = df_transacoes['VL'].values
    valor_medio = np.mean(valores_transacoes)
    valor_std = np.std(valores_transacoes)
    
    # Adicionar arestas baseadas nas transações
    for _, row in df_transacoes.iterrows():
        origem = row['ID_PGTO']
        destino = row['ID_RCBE']
        valor = row['VL']
        
        # Calcular peso normalizado (z-score)
        peso_normalizado = (valor - valor_medio) / valor_std if valor_std > 0 else 0
        
        if G.has_edge(origem, destino):
            # Se já existe aresta, acumular valores e pesos
            G[origem][destino]['valor_total'] += valor
            G[origem][destino]['count'] += 1
            G[origem][destino]['peso_medio'] = G[origem][destino]['valor_total'] / G[origem][destino]['count']
            G[origem][destino]['peso_normalizado'] = (G[origem][destino]['peso_medio'] - valor_medio) / valor_std
        else:
            G.add_edge(origem, destino, 
                      valor_total=valor, 
                      count=1, 
                      peso_medio=valor,
                      peso_normalizado=peso_normalizado)
    
    if len(G.nodes()) == 0:
        return None
    
    # Calcular layout
    pos = nx.spring_layout(G, k=2, iterations=100)
    
    # Preparar dados para plotly - criar traços separados por cor
    edge_traces = []
    
    # Dicionário para agrupar arestas por cor
    edges_by_color = {
        'red': {'x': [], 'y': [], 'width': [], 'info': []},
        'orange': {'x': [], 'y': [], 'width': [], 'info': []},
        'blue': {'x': [], 'y': [], 'width': [], 'info': []},
        'lightblue': {'x': [], 'y': [], 'width': [], 'info': []}
    }
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        valor_total = G[edge[0]][edge[1]]['valor_total']
        count = G[edge[0]][edge[1]]['count']
        peso_normalizado = G[edge[0]][edge[1]]['peso_normalizado']
        
        # Determinar cor baseada no peso
        if peso_normalizado > 1:  # Alto valor
            cor = 'red'
            largura = 4
        elif peso_normalizado > 0:  # Valor médio-alto
            cor = 'orange'
            largura = 3
        elif peso_normalizado > -1:  # Valor médio-baixo
            cor = 'blue'
            largura = 2
        else:  # Baixo valor
            cor = 'lightblue'
            largura = 1
        
        # Adicionar às coordenadas da cor correspondente
        edges_by_color[cor]['x'].extend([x0, x1, None])
        edges_by_color[cor]['y'].extend([y0, y1, None])
        edges_by_color[cor]['width'].append(largura)
        edges_by_color[cor]['info'].append(f"{edge[0]} ↔ {edge[1]}<br>Valor Total: R$ {valor_total:,.0f}<br>Transações: {count}<br>Peso: {peso_normalizado:.2f}")
    
    # Criar traços das arestas para cada cor
    for cor, dados in edges_by_color.items():
        if dados['x']:  # Só criar traço se houver dados
            edge_trace = go.Scatter(
                x=dados['x'], 
                y=dados['y'],
                line=dict(width=dados['width'][0] if dados['width'] else 2, color=cor),
                hoverinfo='none',
                mode='lines',
                name=f"Arestas {cor}"
            )
            edge_traces.append(edge_trace)
    
    # Preparar dados dos nós
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        degree = G.degree(node)
        
        # Determinar cor do nó baseado no tipo de relação
        if cnpj_selecionado and node == cnpj_selecionado:
            cor_no = 'red'
            tamanho = 30
            info_adicional = " (CNPJ Selecionado)"
        else:
            cor_no = 'lightblue'
            tamanho = 15 + degree * 2  # Tamanho baseado no grau
            info_adicional = ""
        
        node_colors.append(cor_no)
        node_sizes.append(tamanho)
        
        node_info.append(f"{node}{info_adicional}<br>Grau: {degree}")
    
    # Criar traço dos nós
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        hovertext=node_info,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='black')
        )
    )
    
    # Criar figura com todos os traços
    all_traces = edge_traces + [node_trace]
    fig = go.Figure(data=all_traces,
                    layout=go.Layout(
                        title=dict(text='Rede de Relacionamentos Comerciais', font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ 
                            dict(
                                text="🔴 CNPJ Selecionado | 🟠 Alto Valor | 🔵 Médio-Alto | 🔷 Médio-Baixo | 🔹 Baixo Valor",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.5, y=-0.1,
                                xanchor='center', yanchor='top',
                                font=dict(color='gray', size=10)
                            )
                        ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                    ))
    
    return fig

def create_saldo_progress_chart(df_infos, cnpj_selecionado):
    """Cria gráfico de progresso do saldo por mês"""
    
    # Filtrar dados do CNPJ selecionado
    dados_cnpj = df_infos[df_infos['ID'] == cnpj_selecionado].copy()
    
    if len(dados_cnpj) == 0:
        return None
    
    # Ordenar por mês
    dados_cnpj = dados_cnpj.sort_values('DT_REFE')
    
    # Criar gráfico de linha com barras
    fig = go.Figure()
    
    # Adicionar linha do saldo
    fig.add_trace(go.Scatter(
        x=dados_cnpj['DT_REFE'],
        y=dados_cnpj['VL_SLDO'],
        mode='lines+markers',
        name='Saldo',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Saldo: R$ %{y:,.0f}<extra></extra>'
    ))
    
    # Calcular deltas (variação mês a mês)
    dados_cnpj['Delta'] = dados_cnpj['VL_SLDO'].diff()
    dados_cnpj['Delta_Percentual'] = (dados_cnpj['VL_SLDO'].pct_change() * 100).round(2)
    
    # Adicionar barras de variação
    cores_delta = []
    for delta in dados_cnpj['Delta']:
        if pd.isna(delta):
            cores_delta.append('gray')
        elif delta > 0:
            cores_delta.append('green')
        else:
            cores_delta.append('red')
    
    fig.add_trace(go.Bar(
        x=dados_cnpj['DT_REFE'],
        y=dados_cnpj['Delta'],
        name='Variação Mensal',
        marker_color=cores_delta,
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>Variação: R$ %{y:,.0f}<extra></extra>',
        yaxis='y2'
    ))
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text=f'📈 Progresso do Saldo - {cnpj_selecionado}',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='Mês',
            tickangle=45
        ),
        yaxis=dict(
            title='Saldo (R$)',
            side='left',
            tickformat='R$ ,.0f'
        ),
        yaxis2=dict(
            title='Variação Mensal (R$)',
            side='right',
            overlaying='y',
            tickformat='R$ ,.0f'
        ),
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Adicionar linha de referência zero para variação
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig, dados_cnpj

def create_heatmap_temporal(df_transacoes):
    """Cria heatmap temporal das transações"""
    
    # Preparar dados
    df_heatmap = df_transacoes.copy()
    df_heatmap['Mes'] = df_heatmap['DT_REFE'].dt.month
    df_heatmap['Dia'] = df_heatmap['DT_REFE'].dt.day
    
    # Agrupar por mês e dia
    heatmap_data = df_heatmap.groupby(['Mes', 'Dia']).size().reset_index(name='Transacoes')
    
    # Criar matriz para heatmap
    meses = ['Março', 'Abril', 'Maio']
    dias = list(range(1, 32))  # Converter range para list
    
    matriz = np.zeros((len(meses), 31))
    
    for _, row in heatmap_data.iterrows():
        mes_idx = row['Mes'] - 3  # Março = 0, Abril = 1, Maio = 2
        dia_idx = row['Dia'] - 1
        if 0 <= mes_idx < 3 and 0 <= dia_idx < 31:
            matriz[mes_idx, dia_idx] = row['Transacoes']
    
    # Criar heatmap
    fig = px.imshow(
        matriz,
        labels=dict(x="Dia do Mês", y="Mês", color="Transações"),
        x=dias,
        y=meses,
        color_continuous_scale='Reds',
        title="Heatmap Temporal - Transações por Dia"
    )
    
    fig.update_layout(height=400)
    return fig

def create_risk_score_chart(df_infos):
    """Cria gráfico de score de risco"""
    
    # Remover duplicatas para análise correta
    df_risk = df_infos.drop_duplicates(subset=['ID']).copy()
    
    # Normalizar saldo (valores negativos = maior risco)
    df_risk['Saldo_Normalizado'] = (df_risk['VL_SLDO'] - df_risk['VL_SLDO'].min()) / (df_risk['VL_SLDO'].max() - df_risk['VL_SLDO'].min())
    
    # Score de risco (0 = baixo risco, 1 = alto risco)
    df_risk['Risk_Score'] = 1 - df_risk['Saldo_Normalizado']
    
    # Categorizar risco
    df_risk['Risk_Category'] = pd.cut(df_risk['Risk_Score'], 
                                    bins=[0, 0.3, 0.7, 1], 
                                    labels=['Baixo Risco', 'Médio Risco', 'Alto Risco'])
    
    # Contar por categoria
    risk_counts = df_risk['Risk_Category'].value_counts()
    
    # Criar gráfico de pizza
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Distribuição de Risco dos CNPJs",
        color_discrete_sequence=['#28a745', '#ffc107', '#dc3545']
    )
    
    return fig

def create_sector_analysis(df_infos, df_transacoes):
    """Análise detalhada por setor"""
    
    # Remover duplicatas para análise correta
    df_unique = df_infos.drop_duplicates(subset=['ID']).copy()
    
    # Merge dos dados
    df_merged = df_unique.merge(
        df_transacoes.groupby('ID_PGTO')['VL'].sum().reset_index(),
        left_on='ID', right_on='ID_PGTO', how='left'
    )
    
    # Análise por setor
    sector_analysis = df_merged.groupby('DS_CNAE').agg({
        'VL_FATU': ['mean', 'std', 'count'],
        'VL_SLDO': ['mean', 'std'],
        'VL': ['sum', 'mean']
    }).round(2)
    
    sector_analysis.columns = [
        'Faturamento_Medio', 'Faturamento_Std', 'Qtd_Empresas',
        'Saldo_Medio', 'Saldo_Std', 'Volume_Total', 'Volume_Medio'
    ]
    
    # Filtrar setores com pelo menos 5 empresas
    sector_analysis = sector_analysis[sector_analysis['Qtd_Empresas'] >= 5]
    
    # Criar scatter plot
    fig = px.scatter(
        sector_analysis,
        x='Faturamento_Medio',
        y='Volume_Total',
        size='Qtd_Empresas',
        color='Saldo_Medio',
        hover_name=sector_analysis.index,
        hover_data=['Faturamento_Std', 'Saldo_Std'],
        title="Análise por Setor: Faturamento vs Volume de Transações",
        color_continuous_scale='RdBu',
        size_max=50
    )
    
    fig.update_layout(height=500)
    return fig

def create_transaction_patterns(df_transacoes):
    """Análise de padrões de transação"""
    
    # Padrões por tipo de transação
    patterns = df_transacoes.groupby(['DS_TRAN', df_transacoes['DT_REFE'].dt.hour]).size().reset_index(name='Count')
    
    fig = px.bar(
        patterns,
        x='DT_REFE',
        y='Count',
        color='DS_TRAN',
        title="Padrões de Transação por Hora do Dia",
        barmode='group'
    )
    
    fig.update_layout(height=400)
    return fig
