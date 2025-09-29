import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar sistema RAG
from rag_system import create_chat_interface

# Importar visualizações avançadas
from visualizacoes_avancadas import (
    create_sankey_diagram,
    create_network_graph,
    create_heatmap_temporal,
    create_risk_score_chart,
    create_sector_analysis,
    create_saldo_progress_chart
)

# Configuração da página
st.set_page_config(
    page_title="Dashboard Santander - Análise de CNPJs",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        color: #e60012;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #e60012;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stMetric {
        font-size: 8px;
    }
    .stMetric [data-testid="metric-container"] {
        font-size: 8px;
    }
    .stMetric [data-testid="metric-value"] {
        font-size: 10px;
    }
    .stMetric [data-testid="metric-label"] {
        font-size: 6px;
    }
    .stMetric [data-testid="metric-delta"] {
        font-size: 6px;
    }
    /* CSS mais específico para métricas principais */
    div[data-testid="metric-container"] {
        font-size: 8px !important;
    }
    div[data-testid="metric-container"] > div {
        font-size: 8px !important;
    }
    div[data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 10px !important;
    }
    div[data-testid="metric-container"] [data-testid="metric-label"] {
        font-size: 6px !important;
    }
    div[data-testid="metric-container"] [data-testid="metric-delta"] {
        font-size: 6px !important;
    }
    .stDataFrame {
        font-size: 8px;
    }
    .stSelectbox {
        font-size: 9px;
    }
    .stSubheader {
        font-size: 12px;
    }
    .stMarkdown h3 {
        font-size: 11px;
    }
    .stMarkdown h4 {
        font-size: 10px;
    }
    .stMarkdown h2 {
        font-size: 13px;
    }
    .stMarkdown h1 {
        font-size: 40px;
    }
    .stExpander {
        font-size: 9px;
    }
</style>
""", unsafe_allow_html=True)

def format_currency(value):
    """Formatar valores monetários com ponto como separador de milhares"""
    if pd.isna(value) or value == 0:
        return "R$ 0"
    return f"R$ {value:,.0f}".replace(",", ".")

def load_data():
    """Carregar dados dos arquivos CSV"""
    try:
        # Carregar dados de informações (usar separador ponto e vírgula)
        df_infos = pd.read_csv('dados/Base_Infos.csv', sep=';')
        
        # Carregar dados de transações (usar separador ponto e vírgula)
        df_transacoes = pd.read_csv('dados/Base_Transacoes.csv', sep=';')
        
        # Carregar dados analisados (usar separador vírgula)
        df_analisada = pd.read_csv('dados/Base_Analisada.csv')
        
        # Converter datas
        df_infos['DT_REFE'] = pd.to_datetime(df_infos['DT_REFE'])
        df_transacoes['DT_REFE'] = pd.to_datetime(df_transacoes['DT_REFE'])
        df_analisada['DT_REFE'] = pd.to_datetime(df_analisada['DT_REFE'])
        
        return df_infos, df_transacoes, df_analisada
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None, None

def calcular_momento_vida(df_infos, cnpj_selecionado):
    """Calcular momento de vida da empresa"""
    dados_cnpj = df_infos[df_infos['ID'] == cnpj_selecionado].copy()
    
    if len(dados_cnpj) == 0:
        return "N/A", 0
    
    # Calcular idade em meses
    idade_meses = len(dados_cnpj)
    
    # Calcular crescimento do saldo
    saldo_inicial = dados_cnpj['VL_SLDO'].iloc[0]
    saldo_final = dados_cnpj['VL_SLDO'].iloc[-1]
    crescimento_saldo = ((saldo_final - saldo_inicial) / abs(saldo_inicial)) * 100 if saldo_inicial != 0 else 0
    
    # Calcular estabilidade (coeficiente de variação)
    cv_saldo = dados_cnpj['VL_SLDO'].std() / dados_cnpj['VL_SLDO'].mean() if dados_cnpj['VL_SLDO'].mean() != 0 else 1
    
    # Classificar momento de vida
    if idade_meses <= 2:
        momento = "Iniciante"
        score = 20
    elif crescimento_saldo > 20 and cv_saldo < 0.3:
        momento = "Desenvolvimento"
        score = 80
    elif crescimento_saldo > 0 and cv_saldo < 0.5:
        momento = "Madura"
        score = 60
    elif crescimento_saldo < -10:
        momento = "Declínio"
        score = 20
    else:
        momento = "Madura"
        score = 50
    
    return momento, score

def calcular_saude_empresa(df_infos, df_transacoes, cnpj_selecionado):
    """Calcular saúde da empresa (0-100)"""
    dados_cnpj = df_infos[df_infos['ID'] == cnpj_selecionado].copy()
    
    if len(dados_cnpj) == 0:
        return 0, "N/A"
    
    # Fatores de saúde
    saldo_atual = dados_cnpj['VL_SLDO'].iloc[-1]
    saldo_inicial = dados_cnpj['VL_SLDO'].iloc[0]
    
    # 1. Crescimento do saldo (30%)
    crescimento = ((saldo_atual - saldo_inicial) / abs(saldo_inicial)) * 100 if saldo_inicial != 0 else 0
    score_crescimento = min(100, max(0, 50 + crescimento * 2))
    
    # 2. Estabilidade (25%)
    cv_saldo = dados_cnpj['VL_SLDO'].std() / dados_cnpj['VL_SLDO'].mean() if dados_cnpj['VL_SLDO'].mean() != 0 else 1
    score_estabilidade = min(100, max(0, 100 - (cv_saldo * 100)))
    
    # 3. Posição do saldo (25%)
    faturamento = dados_cnpj['VL_FATU'].iloc[0]
    ratio_saldo_fatu = (saldo_atual / faturamento) * 100 if faturamento != 0 else 0
    score_posicao = min(100, max(0, ratio_saldo_fatu * 0.5))  # Reduzido de *2 para *0.5
    
    # 4. Tendência mensal (20%)
    if len(dados_cnpj) > 1:
        deltas = dados_cnpj['VL_SLDO'].diff().dropna()
        meses_positivos = len(deltas[deltas > 0])
        score_tendencia = (meses_positivos / len(deltas)) * 100
    else:
        score_tendencia = 50
    
    # Score final ponderado (garantir que não passe de 100)
    score_final = min(100, max(0, score_crescimento * 0.3 + score_estabilidade * 0.25 + 
                   score_posicao * 0.25 + score_tendencia * 0.2))
    
    # Categorizar saúde
    if score_final >= 80:
        categoria = "Alta"
    elif score_final >= 60:
        categoria = "Média"
    else:
        categoria = "Baixa"
    
    return score_final, categoria

def calcular_risco_dependencia(df_transacoes, cnpj_selecionado):
    """Calcular risco de dependência (0-100)"""
    # Transações do CNPJ
    transacoes_cnpj = df_transacoes[
        (df_transacoes['ID_PGTO'] == cnpj_selecionado) | 
        (df_transacoes['ID_RCBE'] == cnpj_selecionado)
    ].copy()
    
    if len(transacoes_cnpj) == 0:
        return 0, "N/A"
    
    # Calcular HHI para concentração
    def calculate_hhi(values):
        if len(values) == 0:
            return 0
        total = sum(values)
        if total == 0:
            return 0
        return sum((v/total)**2 for v in values)
    
    pagamentos = transacoes_cnpj[transacoes_cnpj['ID_PGTO'] == cnpj_selecionado]
    recebimentos = transacoes_cnpj[transacoes_cnpj['ID_RCBE'] == cnpj_selecionado]
    
    hhi_pagamento = calculate_hhi(pagamentos.groupby('ID_RCBE')['VL'].sum().values)
    hhi_recebimento = calculate_hhi(recebimentos.groupby('ID_PGTO')['VL'].sum().values)
    hhi_medio = (hhi_pagamento + hhi_recebimento) / 2
    
    # Calcular número de parceiros
    num_parceiros = len(set(pagamentos['ID_RCBE'].unique()) | set(recebimentos['ID_PGTO'].unique()))
    
    # Calcular volume total
    volume_total = pagamentos['VL'].sum() + recebimentos['VL'].sum()
    
    # Score de risco (0-100, onde 100 = alto risco)
    score_hhi = hhi_medio * 100  # HHI já está entre 0-1
    score_parceiros = min(100, max(0, 100 - (num_parceiros * 3)))  # Ajustado de *5 para *3
    score_volume = min(100, max(0, 100 - (volume_total / 100000)))  # Invertido: volume alto = risco baixo
    
    # Score final (média ponderada) - garantir que não passe de 100
    score_final = min(100, max(0, score_hhi * 0.5 + score_parceiros * 0.3 + score_volume * 0.2))
    
    # Categorizar risco
    if score_final >= 70:
        categoria = "Alto"
    elif score_final >= 40:
        categoria = "Médio"
    else:
        categoria = "Baixo"
    
    return score_final, categoria

def calcular_score_santander(df_infos, df_transacoes, cnpj_selecionado):
    """Calcular Score de Cliente Santander (SCS)"""
    
    def calculate_hhi(values):
        """Calcular Herfindahl-Hirschman Index"""
        if len(values) == 0:
            return 0
        total = sum(values)
        if total == 0:
            return 0
        return sum((v/total)**2 for v in values)
    
    # Dados do CNPJ selecionado
    dados_cnpj = df_infos[df_infos['ID'] == cnpj_selecionado].copy()
    
    if len(dados_cnpj) == 0:
        return 0, {}
    
    # Calcular idade da empresa (assumindo que começou em janeiro)
    idade_meses = len(dados_cnpj)
    
    # Calcular indicadores de liquidez
    saldo_atual = dados_cnpj['VL_SLDO'].iloc[-1]
    saldo_inicial = dados_cnpj['VL_SLDO'].iloc[0]
    variacao_saldo = saldo_atual - saldo_inicial
    
    # Burn rate (variação média mensal negativa)
    deltas = dados_cnpj['VL_SLDO'].diff().dropna()
    burn_rate = abs(deltas[deltas < 0].mean()) if len(deltas[deltas < 0]) > 0 else 0
    
    # Runway
    runway = saldo_atual / burn_rate if burn_rate > 0 else float('inf')
    
    # Análise de relacionamentos
    pagamentos = df_transacoes[df_transacoes['ID_PGTO'] == cnpj_selecionado]
    recebimentos = df_transacoes[df_transacoes['ID_RCBE'] == cnpj_selecionado]
    
    # HHI para concentração
    hhi_pagamento = calculate_hhi(pagamentos.groupby('ID_RCBE')['VL'].sum().values)
    hhi_recebimento = calculate_hhi(recebimentos.groupby('ID_PGTO')['VL'].sum().values)
    hhi_medio = (hhi_pagamento + hhi_recebimento) / 2
    
    # Diversificação
    num_relacionamentos = len(set(pagamentos['ID_RCBE'].unique()) | set(recebimentos['ID_PGTO'].unique()))
    
    # Volume de transações
    volume_total = pagamentos['VL'].sum() + recebimentos['VL'].sum()
    
    # Estabilidade (coeficiente de variação do saldo)
    cv_saldo = dados_cnpj['VL_SLDO'].std() / dados_cnpj['VL_SLDO'].mean() if dados_cnpj['VL_SLDO'].mean() != 0 else 1
    
    # Calcular scores por dimensão (0-20 cada)
    
    # 1. Liquidez (20 pontos)
    if saldo_atual > 0 and runway > 12:
        score_liquidez = 20
    elif saldo_atual > 0 and runway > 6:
        score_liquidez = 15
    elif saldo_atual > 0 and runway > 3:
        score_liquidez = 10
    elif saldo_atual > 0:
        score_liquidez = 5
    else:
        score_liquidez = 0
    
    # 2. Relacionamentos (20 pontos)
    if num_relacionamentos > 20 and hhi_medio < 0.3:
        score_relacionamentos = 20
    elif num_relacionamentos > 15 and hhi_medio < 0.5:
        score_relacionamentos = 15
    elif num_relacionamentos > 10 and hhi_medio < 0.7:
        score_relacionamentos = 10
    elif num_relacionamentos > 5:
        score_relacionamentos = 5
    else:
        score_relacionamentos = 0
    
    # 3. Tendências (20 pontos)
    if variacao_saldo > 0 and cv_saldo < 0.2:
        score_tendencias = 20
    elif variacao_saldo > 0 and cv_saldo < 0.4:
        score_tendencias = 15
    elif variacao_saldo > 0:
        score_tendencias = 10
    elif variacao_saldo > -saldo_atual * 0.1:
        score_tendencias = 5
    else:
        score_tendencias = 0
    
    # 4. Atividade (20 pontos)
    if volume_total > 1000000 and idade_meses >= 4:
        score_atividade = 20
    elif volume_total > 500000 and idade_meses >= 3:
        score_atividade = 15
    elif volume_total > 100000 and idade_meses >= 2:
        score_atividade = 10
    elif volume_total > 0:
        score_atividade = 5
    else:
        score_atividade = 0
    
    # 5. Risco (20 pontos)
    if saldo_atual > 0 and burn_rate < saldo_atual * 0.1 and cv_saldo < 0.3:
        score_risco = 20
    elif saldo_atual > 0 and burn_rate < saldo_atual * 0.2 and cv_saldo < 0.5:
        score_risco = 15
    elif saldo_atual > 0 and burn_rate < saldo_atual * 0.3:
        score_risco = 10
    elif saldo_atual > 0:
        score_risco = 5
    else:
        score_risco = 0
    
    # Score total (0-100)
    score_total = score_liquidez + score_relacionamentos + score_tendencias + score_atividade + score_risco
    
    # Detalhamento
    detalhamento = {
        'Liquidez': score_liquidez,
        'Relacionamentos': score_relacionamentos,
        'Tendências': score_tendencias,
        'Atividade': score_atividade,
        'Risco': score_risco,
        'Total': score_total
    }
    
    return score_total, detalhamento

def main():
    """Função principal"""
    st.markdown('<h1 class="main-header">Dashboard Santander - Análise de CNPJs</h1>', unsafe_allow_html=True)
    
    # Link da apresentação
    st.markdown("""
    <div style="text-align: center; margin: 15px 0;">
        <h2 style="color: #e60012; font-size: 2rem;">
             <a href="https://youtu.be/38opEqyDM8w" target="_blank" style="color: #e60012; text-decoration: none;">
                LINK DA APRESENTAÇÃO
            </a>
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Carregar dados
    df_infos, df_transacoes, df_analisada = load_data()
    
    if df_infos is None or df_transacoes is None or df_analisada is None:
        st.error("Não foi possível carregar os dados. Verifique se os arquivos estão na pasta correta.")
        return
    
    # Sidebar para navegação
    # Logo da Quantum
    st.sidebar.image("Logo-Quantum.png", width=200)
    st.sidebar.markdown("---")
    
    st.sidebar.title("Navegação")
    page = st.sidebar.radio(
        "Escolha uma página:",
        ["Dashboard Geral", "Análise Individual", "Assistente IA", "Equipe Quantum"]
    )
    
    # Navegação entre páginas
    if page == "Dashboard Geral":
        dashboard_geral(df_infos, df_transacoes)
    elif page == "Análise Individual":
        analise_individual(df_infos, df_transacoes, df_analisada)
    elif page == "Assistente IA":
        assistente_ia(df_infos, df_transacoes)
    elif page == "Equipe Quantum":
        equipe_quantum()

def equipe_quantum():
    """Página com informações da equipe Quantum"""
    st.title("Equipe Quantum")
    st.markdown("---")
    
    # Adicionar logo centralizada
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("Logo-Quantum.png", width=300)
    
    st.markdown("---")
    
    # Título da seção
    st.header("Integrantes da Equipe")
    
    # Informações dos membros em colunas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### **Alexandre Ilha de Vilhena**
        **RM:** 88689  
        **Especialidade:** Desenvolvimento e Arquitetura de Sistemas
        
        ---
        
        ### **Erik Hoon Ko**
        **RM:** 93599  
        **Especialidade:** Análise de Dados e Machine Learning
        
        ---
        
        ### **Rafael Fiel Cruz Miranda**
        **RM:** 94654  
        **Especialidade:** Data Science e Visualização
        """)
    
    with col2:
        st.markdown("""
        ### **Luca Moraes Zaharic**
        **RM:** 95794  
        **Especialidade:** Backend e Integração de Sistemas
        
        ---
        
        ### **Bruno Norões de Magalhães**
        **RM:** 82511  
        **Especialidade:** Frontend e UX/UI
        """)

def dashboard_geral(df_infos, df_transacoes):
    """Dashboard geral com visão de todos os CNPJs"""
    
    st.header("Dashboard Geral")
    
    # Carregar dados analisados para o dashboard geral
    df_analisada = pd.read_csv('dados/Base_Analisada.csv')
    df_analisada['DT_REFE'] = pd.to_datetime(df_analisada['DT_REFE'])
    
    # Pegar dados do último mês
    ultimo_mes_data = df_analisada[df_analisada['MES'] == df_analisada['MES'].max()]
    
    # Top Empresas para Empréstimos
    st.subheader("Top Empresas para Empréstimos")
    
    # Dropdown informativo para top empresas para empréstimos
    with st.expander("Sobre as Top Empresas para Empréstimos", expanded=False):
        st.markdown("""
        **Critérios para identificar empresas ideais para empréstimos:**
        - **Saúde alta**: Score de saúde acima do percentil 25
        - **Risco baixo**: Score de risco de concentração abaixo do percentil 75
        - **Runway adequado**: Pelo menos 1 mês de runway
        - **Sem stress**: Flag de stress de caixa = 0
        - **Crescimento positivo**: Crescimento de volume positivo
        - **Maturidade**: Empresas maduras ou em desenvolvimento
        
        **Para o Santander:** Essas empresas são candidatas ideais para produtos de crédito, 
        pois apresentam baixo risco e alta capacidade de pagamento.
        """)
    
    # Filtrar empresas ideais para empréstimos
    empresas_ideais = ultimo_mes_data[
        (ultimo_mes_data['score_saude_model_0_100_lgbm'] >= ultimo_mes_data['q25_health']) &
        (ultimo_mes_data['score_dependencia_risco_0_100'] < ultimo_mes_data['q75_risk']) &
        (ultimo_mes_data['runway_meses'] >= 1) &
        (ultimo_mes_data['flag_stress_caixa'] == 0) &
        (ultimo_mes_data['grow_volume_total_mes'] > 0) &
        (ultimo_mes_data['estado_maturidade_hard'].isin(['Madura', 'Desenvolvimento']))
    ].copy()
    
    if len(empresas_ideais) > 0:
        # Ordenar por score de saúde (maior primeiro)
        empresas_ideais = empresas_ideais.sort_values('score_saude_model_0_100_lgbm', ascending=False)
        
        # Pegar top 10
        top_empresas_ideais = empresas_ideais.head(10)
        
        # Criar gráfico de barras
        fig_top_ideais = px.bar(
            top_empresas_ideais,
            x='score_saude_model_0_100_lgbm',
            y='ID',
            orientation='h',
            title="Top 10 Empresas Ideais para Empréstimos",
            color='score_saude_model_0_100_lgbm',
            color_continuous_scale='Greens',
            labels={'score_saude_model_0_100_lgbm': 'Score de Saúde', 'ID': 'CNPJ'}
        )
        
        fig_top_ideais.update_layout(
            height=500,
            xaxis_title="Score de Saúde (0-100)",
            yaxis_title="CNPJ",
            yaxis={'categoryorder':'total ascending'}
        )
        
        # Adicionar annotations com valores (apenas se não for NaN)
        for i, (cnpj, score) in enumerate(zip(top_empresas_ideais['ID'], top_empresas_ideais['score_saude_model_0_100_lgbm'])):
            if not pd.isna(score):

                fig_top_ideais.add_annotation(
                    x=score + max(top_empresas_ideais['score_saude_model_0_100_lgbm']) * 0.21,

                    y=cnpj,
                    text=f"{score:.1f}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    xanchor='left'
                )
        
        st.plotly_chart(fig_top_ideais, use_container_width=True)
        
        # Tabela detalhada
        st.markdown("#### Detalhes das Top Empresas para Empréstimos")
        
        tabela_ideais = top_empresas_ideais[['ID', 'score_saude_model_0_100_lgbm', 'score_dependencia_risco_0_100', 
                                           'runway_meses', 'grow_volume_total_mes', 'estado_maturidade_hard']].copy()
        
        tabela_ideais.columns = ['CNPJ', 'Saúde', 'Risco Concentração', 'Runway (Meses)', 'Crescimento (%)', 'Estado']
        tabela_ideais['Crescimento (%)'] = (tabela_ideais['Crescimento (%)'] * 100).round(1)
        tabela_ideais['Saúde'] = tabela_ideais['Saúde'].round(1)
        tabela_ideais['Risco Concentração'] = tabela_ideais['Risco Concentração'].round(1)
        tabela_ideais['Runway (Meses)'] = tabela_ideais['Runway (Meses)'].round(1)
        
        st.dataframe(tabela_ideais, use_container_width=True)
        
    else:
        st.info("Nenhuma empresa atende aos critérios ideais para empréstimos no momento.")
    
    # Top Empresas Perigosas
    st.subheader("Top Empresas Perigosas")
    
    # Dropdown informativo para top empresas perigosas
    with st.expander("Sobre as Top Empresas Perigosas", expanded=False):
        st.markdown("""
        **Critérios para identificar empresas de alto risco:**
        - **Saúde baixa**: Score de saúde abaixo do percentil 25
        - **Risco alto**: Score de risco de concentração acima do percentil 75
        - **Runway crítico**: Menos de 3 meses de runway ou já negativo
        - **Com stress**: Flag de stress de caixa = 1
        - **Crescimento negativo**: Crescimento de volume negativo
        - **Declínio**: Empresas em declínio
        
        **Para o Santander:** Essas empresas precisam de atenção especial e podem 
        representar risco de inadimplência. Recomenda-se monitoramento intensivo.
        """)
    
    # Filtrar empresas perigosas
    empresas_perigosas = ultimo_mes_data[
        (ultimo_mes_data['score_saude_model_0_100_lgbm'] <= ultimo_mes_data['q25_health']) &
        (ultimo_mes_data['score_dependencia_risco_0_100'] >= ultimo_mes_data['q75_risk']) &
        ((ultimo_mes_data['runway_meses'] < 3) | (ultimo_mes_data['runway_meses'].isna())) &
        (ultimo_mes_data['flag_stress_caixa'] == 1) &
        (ultimo_mes_data['grow_volume_total_mes'] < 0) &
        (ultimo_mes_data['estado_maturidade_hard'].isin(['Declínio', 'Declínio Persistente']))
    ].copy()
    
    if len(empresas_perigosas) > 0:
        # Ordenar por score de saúde (maior primeiro - igual ao de empréstimos, mas depois inverter)
        empresas_perigosas = empresas_perigosas.sort_values('score_saude_model_0_100_lgbm', ascending=False)
        # Inverter para pegar os piores casos (menores scores)
        empresas_perigosas = empresas_perigosas.tail(10)
        
        # Os top 10 perigosos já foram selecionados com tail(10)
        top_empresas_perigosas = empresas_perigosas.copy()
        
        # Criar gráfico de barras
        fig_top_perigosas = px.bar(
            top_empresas_perigosas,
            x='score_saude_model_0_100_lgbm',
            y='ID',
            orientation='h',
            title="Top 10 Empresas de Alto Risco",
            color='score_saude_model_0_100_lgbm',
            color_continuous_scale='Reds',
            labels={'score_saude_model_0_100_lgbm': 'Score de Saúde', 'ID': 'CNPJ'}
        )
        
        fig_top_perigosas.update_layout(
            height=500,
            xaxis_title="Score de Saúde (0-100)",
            yaxis_title="CNPJ",
            yaxis={'categoryorder':'total descending'}  # Ordem decrescente para mostrar menores valores primeiro
        )
        
        # Adicionar annotations com valores (apenas se não for NaN)
        for i, (cnpj, score) in enumerate(zip(top_empresas_perigosas['ID'], top_empresas_perigosas['score_saude_model_0_100_lgbm'])):
            if not pd.isna(score):
                fig_top_perigosas.add_annotation(
                    x=score + max(top_empresas_perigosas['score_saude_model_0_100_lgbm']) * 0.01,
                    y=cnpj,
                    text=f"{score:.1f}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    xanchor='left'
                )
        
        st.plotly_chart(fig_top_perigosas, use_container_width=True)
        
        # Tabela detalhada
        st.markdown("#### Detalhes das Top Empresas Perigosas")
        
        tabela_perigosas = top_empresas_perigosas[['ID', 'score_saude_model_0_100_lgbm', 'score_dependencia_risco_0_100', 
        'runway_meses', 'grow_volume_total_mes', 'estado_maturidade_hard']].copy()
        
        tabela_perigosas.columns = ['CNPJ', 'Saúde', 'Risco Concentração', 'Runway (Meses)', 'Crescimento (%)', 'Estado']
        tabela_perigosas['Crescimento (%)'] = (tabela_perigosas['Crescimento (%)'] * 100).round(1)
        tabela_perigosas['Saúde'] = tabela_perigosas['Saúde'].round(1)
        tabela_perigosas['Risco Concentração'] = tabela_perigosas['Risco Concentração'].round(1)
        
        # Tratar runway negativo ou NaN
        tabela_perigosas['Runway (Meses)'] = tabela_perigosas['Runway (Meses)'].fillna(0)
        tabela_perigosas['Runway (Meses)'] = tabela_perigosas['Runway (Meses)'].round(1)
        
        # Ordenar tabela pelo score de saúde (menores primeiro)
        tabela_perigosas = tabela_perigosas.sort_values('Saúde', ascending=True)
        
        st.dataframe(tabela_perigosas, use_container_width=True)
        
    else:
        st.info("Nenhuma empresa atende aos critérios de alto risco no momento.")
    
    # Visualizações avançadas
    st.divider()
    
    # Análise de Mudanças Mensais

    st.subheader("Análise de Mudanças Mensais")
    
    # Dropdown informativo para mudanças mensais
    with st.expander("Sobre a Análise de Mudanças Mensais", expanded=False):
        st.markdown("""
        **O que observar:**
        - **Evolução temporal**: Como os indicadores mudam mês a mês
        - **Tendências da carteira**: Saúde média, risco de concentração médio, empresas em stress
        - **Distribuição por estado**: Quantas empresas em cada estado de maturidade
        - **Indicadores críticos**: Runway, volatilidade, crescimento
        
        **Para o Santander:** Identifica tendências gerais da carteira e momentos 
        críticos que podem impactar o negócio.
        """)
    
    # Carregar dados analisados para o dashboard geral
    df_analisada = pd.read_csv('dados/Base_Analisada.csv')
    df_analisada['DT_REFE'] = pd.to_datetime(df_analisada['DT_REFE'])
    
    # Análise de Evolução Temporal dos Indicadores
    st.subheader("Evolução Temporal dos Indicadores")
    
    # Calcular estatísticas mensais usando dados pré-calculados
    stats_mensais = df_analisada.groupby('MES').agg({
        'score_saude_model_0_100': ['mean', 'std', 'count'],
        'score_dependencia_risco_0_100': ['mean', 'std'],
        'runway_meses': ['mean', 'median'],
        'flag_stress_caixa': 'sum',
        'VL_SLDO': lambda x: (x < 0).sum(),  # Contar saldos negativos
        'estado_maturidade_hard': lambda x: x.value_counts().to_dict(),  # Contar por estado
        'grow_volume_total_mes': 'mean',
        'q75_volat': 'mean'
    }).round(2)
    
    # Flatten column names
    stats_mensais.columns = ['Saude_Media', 'Saude_Std', 'Total_Empresas', 'Risco_Media', 'Risco_Std', 
                            'Runway_Media', 'Runway_Mediana', 'Empresas_Stress', 'Empresas_Saldo_Negativo',
                            'Estados_Maturidade', 'Crescimento_Volume_Medio', 'Volatilidade_Media']
    
    stats_mensais = stats_mensais.reset_index()
    
    # Gráfico 1: Evolução da Saúde e Risco
    col1, col2 = st.columns(2)
    
    with col1:
        fig_saude_evolucao = go.Figure()
        
        fig_saude_evolucao.add_trace(go.Scatter(
            x=stats_mensais['MES'],
            y=stats_mensais['Saude_Media'],
            mode='lines+markers',
            name='Saúde Média',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        # Adicionar banda de desvio padrão
        fig_saude_evolucao.add_trace(go.Scatter(
            x=stats_mensais['MES'],
            y=stats_mensais['Saude_Media'] + stats_mensais['Saude_Std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_saude_evolucao.add_trace(go.Scatter(
            x=stats_mensais['MES'],
            y=stats_mensais['Saude_Media'] - stats_mensais['Saude_Std'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            name='Desvio Padrão',
            hoverinfo='skip'
        ))
        
        fig_saude_evolucao.update_layout(
            title="Evolução da Saúde Média da Carteira",
            xaxis_title="Mês",
            yaxis_title="Score de Saúde (0-100)",
            height=400
        )
        
        # Adicionar annotations com valores (apenas se não for NaN)
        for i, (mes, valor) in enumerate(zip(stats_mensais['MES'], stats_mensais['Saude_Media'])):
            if not pd.isna(valor):
                fig_saude_evolucao.add_annotation(
                    x=mes,
                    y=valor + max(stats_mensais['Saude_Media']) * 0.02,
                    text=f"{valor:.1f}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    yanchor='bottom'
                )
        
        st.plotly_chart(fig_saude_evolucao, use_container_width=True)
    
    with col2:
        fig_risco_evolucao = go.Figure()
        
        fig_risco_evolucao.add_trace(go.Scatter(
            x=stats_mensais['MES'],
            y=stats_mensais['Risco_Media'],
            mode='lines+markers',
            name='Risco de ConcentraçãoMédio',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig_risco_evolucao.add_trace(go.Scatter(
            x=stats_mensais['MES'],
            y=stats_mensais['Risco_Media'] + stats_mensais['Risco_Std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_risco_evolucao.add_trace(go.Scatter(
            x=stats_mensais['MES'],
            y=stats_mensais['Risco_Media'] - stats_mensais['Risco_Std'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            name='Desvio Padrão',
            hoverinfo='skip'
        ))
        
        fig_risco_evolucao.update_layout(
            title="Evolução do Risco de Concentração Médio da Carteira",
            xaxis_title="Mês",
            yaxis_title="Score de Risco de Concentração (0-100)",
            height=400
        )
        
        # Adicionar annotations com valores (apenas se não for NaN)
        for i, (mes, valor) in enumerate(zip(stats_mensais['MES'], stats_mensais['Risco_Media'])):
            if not pd.isna(valor):
                fig_risco_evolucao.add_annotation(
                    x=mes,
                    y=valor + max(stats_mensais['Risco_Media']) * 0.02,
                    text=f"{valor:.1f}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    yanchor='bottom'
                )
        
        st.plotly_chart(fig_risco_evolucao, use_container_width=True)
    
    # Gráfico 2: Empresas em Situação Crítica
    col1, col2 = st.columns(2)
    
    with col1:
        fig_stress_evolucao = go.Figure()
        
        fig_stress_evolucao.add_trace(go.Scatter(
            x=stats_mensais['MES'],
            y=stats_mensais['Empresas_Stress'],
            mode='lines+markers',
            name='Empresas com Stress',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig_stress_evolucao.add_trace(go.Scatter(
            x=stats_mensais['MES'],
            y=stats_mensais['Empresas_Saldo_Negativo'],
            mode='lines+markers',
            name='Empresas com Saldo Negativo',
            line=dict(color='orange', width=3),
            marker=dict(size=8)
        ))
        
        fig_stress_evolucao.update_layout(
            title="Evolução de Empresas em Situação Crítica",
            xaxis_title="Mês",
            yaxis_title="Número de Empresas",
            height=400
        )
        
        # Adicionar annotations com valores separados por linha (apenas se não for NaN)
        max_valor = max(max(stats_mensais['Empresas_Stress']), max(stats_mensais['Empresas_Saldo_Negativo']))
        for i, (mes, stress, saldo_neg) in enumerate(zip(stats_mensais['MES'], stats_mensais['Empresas_Stress'], stats_mensais['Empresas_Saldo_Negativo'])):
            if not pd.isna(stress) and not pd.isna(saldo_neg):
                fig_stress_evolucao.add_annotation(
                    x=mes,
                    y=max(stress, saldo_neg) + max_valor * 0.02,
                    text=f"S: {stress}<br>N: {saldo_neg}",
                    showarrow=False,
                    font=dict(size=11, color="black"),
                    yanchor='bottom'
                )
        
        st.plotly_chart(fig_stress_evolucao, use_container_width=True)
    
    with col2:
        fig_runway_evolucao = go.Figure()
        
        fig_runway_evolucao.add_trace(go.Scatter(
            x=stats_mensais['MES'],
            y=stats_mensais['Runway_Media'],
            mode='lines+markers',
            name='Runway Médio',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig_runway_evolucao.add_trace(go.Scatter(
            x=stats_mensais['MES'],
            y=stats_mensais['Runway_Mediana'],
            mode='lines+markers',
            name='Runway Mediano',
            line=dict(color='purple', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig_runway_evolucao.update_layout(
            title="Evolução do Runway da Carteira",
            xaxis_title="Mês",
            yaxis_title="Runway (Meses)",
            height=400
        )
        
        # Adicionar annotations com setas próximas aos pontos (apenas se não for NaN)
        for i, (mes, media, mediana) in enumerate(zip(stats_mensais['MES'], stats_mensais['Runway_Media'], stats_mensais['Runway_Mediana'])):
            # Annotation para média
            if not pd.isna(media):
                fig_runway_evolucao.add_annotation(
                    x=mes,
                    y=media,
                    text=f"M: {media:.1f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="blue",
                    ax=0,
                    ay=-15,
                    font=dict(size=11, color="blue"),
                    bgcolor="white",
                    bordercolor="blue",
                    borderwidth=1
                )
            
            # Annotation para mediana
            if not pd.isna(mediana):
                fig_runway_evolucao.add_annotation(
                    x=mes,
                    y=mediana,
                    text=f"Md: {mediana:.1f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="purple",
                    ax=0,
                    ay=15,
                    font=dict(size=11, color="purple"),
                    bgcolor="white",
                    bordercolor="purple",
                    borderwidth=1
                )
        
        st.plotly_chart(fig_runway_evolucao, use_container_width=True)
    
    st.divider()

    # Análise de Distribuição por Estado de Maturidade
    st.subheader("Distribuição por Estado de Maturidade")
    
    # Calcular distribuição mensal por estado
    distribuicao_mensal = []
    for _, row in stats_mensais.iterrows():
        mes = row['MES']
        estados = row['Estados_Maturidade']
        
        for estado, quantidade in estados.items():
            distribuicao_mensal.append({
                'Mes': mes,
                'Estado': estado,
                'Quantidade': quantidade
            })
    
    df_distribuicao = pd.DataFrame(distribuicao_mensal)
    
    if len(df_distribuicao) > 0:
        fig_distribuicao = px.bar(
            df_distribuicao,
            x='Mes',
            y='Quantidade',
            color='Estado',
            title="Distribuição de Empresas por Estado de Maturidade",
            labels={'Quantidade': 'Número de Empresas', 'Mes': 'Mês'},
            color_discrete_map={
                'Madura': '#2ca02c',
                'Desenvolvimento': '#ff7f0e',
                'Declínio': '#d62728'
            }
        )
        
        fig_distribuicao.update_layout(
            height=500,
            xaxis_title="Mês",
            yaxis_title="Número de Empresas",
            legend_title="Estado de Maturidade"
        )
        
        # Configurar para mostrar valores nas barras
        fig_distribuicao.update_traces(
            texttemplate='%{y}',
            textposition='inside',
            textfont=dict(size=12, color='white', family='Arial Black')
        )
        
        st.plotly_chart(fig_distribuicao, use_container_width=True)
    
    st.divider()
 


def assistente_ia(df_infos, df_transacoes):
    """Página do Assistente IA com sistema RAG"""
    
    st.header("Assistente de Análise Santander")
    
    # Carregar dados analisados
    df_analisada = pd.read_csv('dados/Base_Analisada.csv')
    df_analisada['DT_REFE'] = pd.to_datetime(df_analisada['DT_REFE'])
    
    # Armazenar dados no session state para o RAG
    st.session_state.df_infos = df_infos
    st.session_state.df_transacoes = df_transacoes
    st.session_state.df_analisada = df_analisada
    
    # Explicação sobre o assistente
    st.markdown("""
    ### Como funciona o Assistente IA
    
    O **Assistente de Análise Santander** é um sistema inteligente que combina:
    
    - **IA Generativa (ChatGPT-4)**: Para respostas naturais e insights avançados
    - **RAG (Retrieval-Augmented Generation)**: Acesso inteligente aos dados dos CNPJs
    - **Contexto Específico**: Análise baseada nos dados reais da carteira
    
    **Você pode perguntar sobre:**
    - Análises gerais da carteira
    - Insights específicos de CNPJs
    - Identificação de riscos
    - Oportunidades de negócio
    - Comparações e rankings
    """)
    
    # Interface do chat
    create_chat_interface()

def analise_individual(df_infos, df_transacoes, df_analisada):
    """Análise individual de CNPJ específico"""
    
    st.header("Análise Individual")
    
    # Seleção de CNPJ
    cnpjs_disponiveis = sorted(df_infos['ID'].unique())
    cnpj_selecionado = st.selectbox(
        "Selecione um CNPJ para análise:",
        cnpjs_disponiveis,
        key="cnpj_selector"
    )
    
    # Armazenar CNPJ selecionado no session state para o RAG
    st.session_state.cnpj_selecionado = cnpj_selecionado
    
    if not cnpj_selecionado:
        st.warning("Selecione um CNPJ para continuar.")
        return
    
    # Dados do CNPJ selecionado
    dados_cnpj = df_infos[df_infos['ID'] == cnpj_selecionado].copy()
    dados_analisados_cnpj = df_analisada[df_analisada['ID'] == cnpj_selecionado].copy()
    transacoes_cnpj = df_transacoes[
        (df_transacoes['ID_PGTO'] == cnpj_selecionado) | 
        (df_transacoes['ID_RCBE'] == cnpj_selecionado)
    ].copy()
    
    if len(dados_cnpj) == 0:
        st.error("CNPJ não encontrado nos dados.")
        return

    # Insights da Quantum
    st.subheader(f"Insights da Quantum")
    
    if len(dados_analisados_cnpj) > 0:
        ultimo_mes = dados_analisados_cnpj.iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            estado = ultimo_mes['estado_maturidade_hard']
            # st.metric(
            #     "Maturidade da Empresa", 
            #     estado
            # )

            st.markdown(f"**Maturidade da Empresa:**<br/>{estado}", unsafe_allow_html=True)

            with st.expander("Sobre a Maturidade da Empresa", expanded=True):
                if 'explicativo_maturidade_aberto' in ultimo_mes and pd.notna(ultimo_mes['explicativo_maturidade_aberto']):
                    st.info(ultimo_mes['explicativo_maturidade_aberto'])
        
        with col2:
            # Determinar recomendação baseada no estado de maturidade
            estado = ultimo_mes['estado_maturidade_hard']
            
            if estado == "Iniciante":
                recomendacao = "Crédito Inicial Condicionado a Garantias Sólidas e Análise do Plano de Negócios"
            elif estado == "Estagnação":
                recomendacao = "Linha de Curto Prazo para Readequação Estratégica, com Garantias Robustas"
            elif estado == "Desenvolvimento":
                recomendacao = "Capital de Giro para Aceleração e Antecipação de Recebíveis"
            elif estado == "Amadurecimento":
                recomendacao = "Crédito Estruturado para Superação do Platô de Crescimento"
            elif estado == "Madura":
                recomendacao = "Crédito Estratégico de Baixo Custo"
            elif estado == "Expansão":
                recomendacao = "Financiamento para Projetos de Expansão (CAPEX e M&A)"
            elif estado == "Declínio":
                recomendacao = "Não Recomendar Novo Crédito; Foco na Gestão da Exposição Atual"
            elif estado == "Retomada":
                recomendacao = "Crédito Especializado para Reestruturação, Mediante Análise do Plano de Viabilidade"
            elif estado == "Declínio Persistente":
                recomendacao = "Não Recomendar Novo Crédito; Foco na Gestão da Exposição Atual"
            else:
                recomendacao = "Análise Adicional Necessária"
            
            st.markdown(f"**Recomendação de Crédito:**<br/>{recomendacao}", unsafe_allow_html=True)
    
    # Informações básicas
    st.subheader(f"Informações da Empresa: {cnpj_selecionado}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        saldo_atual = dados_cnpj['VL_SLDO'].iloc[-1]
        st.metric("Saldo Atual", format_currency(saldo_atual))
    
    with col2:
        faturamento = dados_cnpj['VL_FATU'].iloc[0]  # Faturamento é igual para todos os meses
        st.metric("Faturamento", format_currency(faturamento))
    
    with col3:
        setor = dados_cnpj['DS_CNAE'].iloc[0]
        st.markdown(f"**Setor:**<br/>{setor}", unsafe_allow_html=True)
    
    with col4:
        idade_meses = len(dados_cnpj)
        st.metric("Idade (Meses)", idade_meses)
    
    # Análise de Evolução da Saúde e Risco
    st.subheader("Evolução da Saúde e Risco")
    
    # Dropdown informativo para evolução da saúde e risco
    with st.expander("Sobre a Evolução da Saúde e Risco", expanded=False):
        st.markdown("""
            Como funciona a Saúde da Empresa
            **Score de Saúde (0-100):**
            - **Alta**: Score ≥ P75 (percentil 75)
            - **Média**: Score entre P25 e P75
            - **Baixa**: Score ≤ P25 (percentil 25)
            **Baseado em:** Crescimento, estabilidade, posição financeira e tendências.

            Como funciona o Risco de Dependência
            **Score de Risco (0-100):**
            - **Baixo**: Score < 40 (bem diversificado)
            - **Médio**: Score 40-69 (concentração moderada)
            - **Alto**: Score ≥ P75 (alta dependência)   
            **Baseado em:** Concentração de relacionamentos, diversificação e volume.

            Obs: Cada variável é calculada mês a mês para mostrar tendências.
            """)

    
    if len(dados_analisados_cnpj) > 0:
        # Ordenar dados por mês
        dados_analisados_cnpj = dados_analisados_cnpj.sort_values('MES')
        
        # Exibir métricas atuais (último mês)
        ultimo_mes = dados_analisados_cnpj.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            saude_atual = ultimo_mes['score_saude_model_0_100_lgbm']
            # Determinar categoria baseada nos percentis
            saude_atual = saude_atual * 10
            if saude_atual >= ultimo_mes['q75_health']:
                saude_cat = "Alta"
            elif saude_atual >= ultimo_mes['q25_health']:
                saude_cat = "Média"
            else:
                saude_cat = "Baixa"
            #como estava antes
            #st.metric("Saúde da Empresa", f"{saude_atual:.2f}", delta=f"{saude_cat}")
            st.metric("Saúde da Empresa", f"{saude_atual / 10:.2f}")
        
        with col2:
            risco_atual = ultimo_mes['score_dependencia_risco_0_100']
            # Determinar categoria baseada no percentil 75
            if risco_atual >= ultimo_mes['q75_risk']:
                risco_cat = "Concentrado"
            else:
                risco_cat = "Não concentrado"
            #como estava antes
            #st.metric("Risco Dependência", f"{risco_atual:.0f}", delta=f"{risco_cat}")
            st.metric("Risco Dependência", f"{risco_atual:.0f}")
        
        with col3:
            # Usar grow_volume_total_mes para tendência
            crescimento_volume = ultimo_mes['grow_volume_total_mes']
            if crescimento_volume > 0.1:  # Crescimento > 10%
                tendencia = "Em crescimento"
            elif crescimento_volume < -0.1:  # Declínio > 10%
                tendencia = "Em declínio"
            else:
                tendencia = "Estável"
            st.metric("Tendência Volume", tendencia)

        # Gráficos de evolução
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de Saúde da Empresa
            fig_saude = go.Figure()
            
            fig_saude.add_trace(go.Scatter(
                x=dados_analisados_cnpj['MES'],
                y=dados_analisados_cnpj['score_saude_model_0_100_lgbm'],
                mode='lines+markers',
                name='Saúde da Empresa',
                line=dict(color='green', width=3),
                marker=dict(size=8)
            ))
            
            # Adicionar linha de referência para percentis
            fig_saude.add_hline(y=ultimo_mes['q75_health'], line_dash="dash", line_color="lightgreen", 
                              annotation_text="P75")
            fig_saude.add_hline(y=ultimo_mes['q25_health'], line_dash="dash", line_color="orange", 
                              annotation_text="P25")
            
            fig_saude.update_layout(
                title="Evolução da Saúde da Empresa",
                xaxis_title="Mês",
                yaxis_title="Score (0-100)",
                height=400
            )
            fig_saude.update_yaxes(range=[0, 100])
            
            # Adicionar annotations com valores (apenas se não for NaN)
            for i, (mes, valor) in enumerate(zip(dados_analisados_cnpj['MES'], dados_analisados_cnpj['score_saude_model_0_100'])):
                if not pd.isna(valor):
                    fig_saude.add_annotation(
                        x=mes,
                        y=valor + 5,
                        text=f"{valor:.1f}",
                        showarrow=False,
                        font=dict(size=11, color="black"),
                        yanchor='bottom'
                    )
            
            st.plotly_chart(fig_saude, use_container_width=True)
        
        with col2:
            # Gráfico de Risco de Dependência
            fig_risco = go.Figure()
            
            fig_risco.add_trace(go.Scatter(
                x=dados_analisados_cnpj['MES'],
                y=dados_analisados_cnpj['score_dependencia_risco_0_100'],
                mode='lines+markers',
                name='Risco de Dependência',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            
            # Adicionar linha de referência para percentil 75
            fig_risco.add_hline(y=ultimo_mes['q75_risk'], line_dash="dash", line_color="orange", 
                              annotation_text="P75")
            
            fig_risco.update_layout(
                title="Evolução do Risco de Dependência",
                xaxis_title="Mês",
                yaxis_title="Score (0-100)",
                height=400
            )
            fig_risco.update_yaxes(range=[0, 100])
            
            # Adicionar annotations com valores (apenas se não for NaN)
            for i, (mes, valor) in enumerate(zip(dados_analisados_cnpj['MES'], dados_analisados_cnpj['score_dependencia_risco_0_100'])):
                if not pd.isna(valor):
                    fig_risco.add_annotation(
                        x=mes,
                        y=valor + 5,
                        text=f"{valor:.1f}",
                        showarrow=False,
                        font=dict(size=11, color="black"),
                        yanchor='bottom'
                    )
            
            st.plotly_chart(fig_risco, use_container_width=True)
        
    
    else:
        st.info("Dados analisados não encontrados para este CNPJ.")
    
    # Análise de fluxo de caixa
    st.subheader("Análise de Fluxo de Caixa")
    
    # Dropdown informativo para fluxo de caixa
    with st.expander("Sobre o Fluxo de Caixa", expanded=False):
        st.markdown("""
        **O que observar:**
        - **Tendência**: Saldo crescendo ou diminuindo ao longo do tempo
        - **Sazonalidade**: Padrões mensais de entrada e saída
        - **Estabilidade**: Variações grandes ou pequenas no saldo
        - **Pontos críticos**: Meses com quedas significativas
        
        **Para o Santander:** Fluxo positivo e estável indica boa saúde financeira. 
        Variações grandes podem indicar necessidade de crédito sazonal.
        """)
    
    if len(dados_cnpj) > 1:
        dados_cnpj_sorted = dados_cnpj.sort_values('DT_REFE')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dados_cnpj_sorted['DT_REFE'],
            y=dados_cnpj_sorted['VL_SLDO'],
            mode='lines+markers',
            name='Saldo',
            line=dict(color='#e60012', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Evolução do Saldo",
            xaxis_title="Data",
            yaxis_title="Saldo (R$)",
            height=400
        )
        fig.update_yaxes(tickformat='R$ ,.0f')
        
        # Adicionar annotations com valores formatados (apenas se não for NaN)
        for i, (data, saldo) in enumerate(zip(dados_cnpj_sorted['DT_REFE'], dados_cnpj_sorted['VL_SLDO'])):
            if not pd.isna(saldo):
                fig.add_annotation(
                    x=data,
                    y=saldo + max(dados_cnpj_sorted['VL_SLDO']) * 0.02,
                    text=f"R$ {saldo:,.0f}".replace(",", "."),
                    showarrow=False,
                    font=dict(size=11, color="black"),
                    yanchor='bottom'
                )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Rede de relacionamentos
    st.subheader("Rede de Relacionamentos")
    
    # Dropdown informativo para rede de relacionamentos
    with st.expander("Sobre a Rede de Relacionamentos", expanded=False):
        st.markdown("""
        **O que observar:**
        - **Conectividade**: Quantos relacionamentos comerciais a empresa tem
        - **Volume**: Valor total das transações com cada parceiro
        - **Concentração**: Poucos parceiros grandes vs muitos pequenos
        - **Diversificação**: Relacionamentos em diferentes setores
        
        **Para o Santander:** Empresas bem conectadas têm menor risco de concentração. 
        Diversificação de parceiros indica estabilidade comercial.
        """)
    
    # Análise de pagamentos e recebimentos
    pagamentos = transacoes_cnpj[transacoes_cnpj['ID_PGTO'] == cnpj_selecionado]
    recebimentos = transacoes_cnpj[transacoes_cnpj['ID_RCBE'] == cnpj_selecionado]
    
    if len(pagamentos) > 0 or len(recebimentos) > 0:
        # Calcular estatísticas de relacionamentos
        num_pagamentos = len(pagamentos['ID_RCBE'].unique())
        num_recebimentos = len(recebimentos['ID_PGTO'].unique())
        volume_pagamentos = pagamentos['VL'].sum()
        volume_recebimentos = recebimentos['VL'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Parceiros (Pagamentos)", num_pagamentos)
        
        with col2:
            st.metric("Parceiros (Recebimentos)", num_recebimentos)
        
        with col3:
            st.metric("Volume Pagamentos", format_currency(volume_pagamentos))
        
        with col4:
            st.metric("Volume Recebimentos", format_currency(volume_recebimentos))
        
        # Calcular HHI para concentração
        def calculate_hhi(values):
            if len(values) == 0:
                return 0
            total = sum(values)
            if total == 0:
                return 0
            return sum((v/total)**2 for v in values)
        
        hhi_pagamento = calculate_hhi(pagamentos.groupby('ID_RCBE')['VL'].sum().values)
        hhi_recebimento = calculate_hhi(recebimentos.groupby('ID_PGTO')['VL'].sum().values)
        hhi_medio = (hhi_pagamento + hhi_recebimento) / 2
        
        # Exibir HHI de forma destacada
        st.markdown("#### Índice de Concentração (HHI)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("HHI Pagamentos", f"{hhi_pagamento:.3f}")
        
        with col2:
            st.metric("HHI Recebimentos", f"{hhi_recebimento:.3f}")
        
        with col3:
            st.metric("HHI Médio", f"{hhi_medio:.3f}")
        
        # Explicação do HHI
        with st.expander("Como funciona o Índice de Concentração (HHI)", expanded=False):
            st.markdown("""
            **O que é o HHI:**
            - **Herfindahl-Hirschman Index**: Mede concentração de relacionamentos
            - **Escala**: 0 (perfeitamente diversificado) a 1 (totalmente concentrado)
            - **Interpretação**: 
              - 0.0-0.3: Baixa concentração (bom)
              - 0.3-0.6: Concentração moderada
              - 0.6-1.0: Alta concentração (risco)
            
            **Para o Santander:** HHI baixo indica empresa diversificada e menos dependente 
            de poucos clientes/fornecedores. HHI alto indica risco de concentração.
            """)
        
        # Tabelas de relacionamentos
        if len(pagamentos) > 0:
            st.markdown("#### Relações de Pagamento")
            
            # Calcular estatísticas por parceiro
            pagamentos_stats = pagamentos.groupby('ID_RCBE').agg({
                'VL': ['sum', 'count', 'mean'],
                'DS_TRAN': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
            }).round(2)
            
            pagamentos_stats.columns = ['Valor_Total', 'Qtd_Transacoes', 'Valor_Medio', 'Forma_Mais_Usada']
            pagamentos_stats = pagamentos_stats.sort_values('Valor_Total', ascending=False)
            
            # Calcular z-score para importância
            valores = pagamentos_stats['Valor_Total'].values
            if len(valores) > 1:
                z_scores = (valores - np.mean(valores)) / np.std(valores)
                pagamentos_stats['Importancia_ZScore'] = z_scores.round(2)
            else:
                pagamentos_stats['Importancia_ZScore'] = 0
            
            # Preparar dados para exibição
            pagamentos_display = pagamentos_stats.copy()
            pagamentos_display['Valor_Total'] = pagamentos_display['Valor_Total'].apply(format_currency)
            pagamentos_display['Valor_Medio'] = pagamentos_display['Valor_Medio'].apply(format_currency)
            pagamentos_display = pagamentos_display.reset_index()
            pagamentos_display.columns = ['CNPJ_Parceiro', 'Valor_Total', 'Qtd_Transacoes', 'Valor_Medio', 'Forma_Mais_Usada', 'Importancia_ZScore']
            
            st.dataframe(pagamentos_display, use_container_width=True)
        
        if len(recebimentos) > 0:
            st.markdown("#### Relações de Recebimento")
            
            # Calcular estatísticas por parceiro
            recebimentos_stats = recebimentos.groupby('ID_PGTO').agg({
                'VL': ['sum', 'count', 'mean'],
                'DS_TRAN': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
            }).round(2)
            
            recebimentos_stats.columns = ['Valor_Total', 'Qtd_Transacoes', 'Valor_Medio', 'Forma_Mais_Usada']
            recebimentos_stats = recebimentos_stats.sort_values('Valor_Total', ascending=False)
            
            # Calcular z-score para importância
            valores = recebimentos_stats['Valor_Total'].values
            if len(valores) > 1:
                z_scores = (valores - np.mean(valores)) / np.std(valores)
                recebimentos_stats['Importancia_ZScore'] = z_scores.round(2)
            else:
                recebimentos_stats['Importancia_ZScore'] = 0
            
            # Preparar dados para exibição
            recebimentos_display = recebimentos_stats.copy()
            recebimentos_display['Valor_Total'] = recebimentos_display['Valor_Total'].apply(format_currency)
            recebimentos_display['Valor_Medio'] = recebimentos_display['Valor_Medio'].apply(format_currency)
            recebimentos_display = recebimentos_display.reset_index()
            recebimentos_display.columns = ['CNPJ_Parceiro', 'Valor_Total', 'Qtd_Transacoes', 'Valor_Medio', 'Forma_Mais_Usada', 'Importancia_ZScore']
            
            st.dataframe(recebimentos_display, use_container_width=True)
        
        # Explicação do Z-Score
        with st.expander("Como funciona o Sistema de Importância (Z-Score)", expanded=False):
            st.markdown("""
            **O que é o Z-Score:**
            - **Z-Score**: Mede quantos desvios padrão um valor está da média
            - **Escala**: Valores positivos = acima da média, negativos = abaixo
            - **Interpretação**:
              - Z-Score > 1: Relacionamento muito importante (top 16%)
              - Z-Score 0-1: Relacionamento importante (média-alta)
              - Z-Score -1-0: Relacionamento normal (média-baixa)
              - Z-Score < -1: Relacionamento pouco importante (bottom 16%)
            
            **Para o Santander:** Z-Score alto indica relacionamentos comerciais mais 
            estratégicos para crédito e investimento.
            """)
    
    else:
        st.info("Nenhuma transação encontrada para este CNPJ.")
    
    
    # Progresso do Saldo
    st.subheader("Progresso do Saldo")
    
    # Dropdown informativo para progresso do saldo
    with st.expander("Sobre o Progresso do Saldo", expanded=False):
        st.markdown("""
        **O que observar:**
        - **Tendência**: Saldo crescendo ou diminuindo ao longo dos meses
        - **Variações**: Meses com grandes mudanças (positivas ou negativas)
        - **Estabilidade**: Padrão consistente ou volátil
        - **Pontos críticos**: Meses com quedas significativas
        
        **Para o Santander:** Progresso positivo indica boa gestão financeira. 
        Variações grandes podem indicar necessidade de crédito ou oportunidades de investimento.
        """)
    
    if len(dados_cnpj) > 0:
        saldo_fig, dados_saldo = create_saldo_progress_chart(df_infos, cnpj_selecionado)
        st.plotly_chart(saldo_fig, use_container_width=True)
        
        # Métricas de resumo do progresso
        if len(dados_saldo) > 0:
            saldo_inicial = dados_saldo['VL_SLDO'].iloc[0]
            saldo_final = dados_saldo['VL_SLDO'].iloc[-1]
            variacao_total = saldo_final - saldo_inicial
            meses_positivos = len(dados_saldo[dados_saldo['Delta'] > 0])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Saldo Inicial", format_currency(saldo_inicial))
            
            with col2:
                st.metric("Saldo Final", format_currency(saldo_final))
            
            with col3:
                st.metric("Variação Total", format_currency(variacao_total))
            
            with col4:
                st.metric("Meses Positivos", f"{meses_positivos}/{len(dados_saldo)}")
    
    # Indicadores de Liquidez e Risco
    st.subheader("Indicadores de Liquidez e Risco")
    
    # Dropdown informativo para indicadores de liquidez
    with st.expander("Sobre os Indicadores de Liquidez", expanded=False):
        st.markdown("""
        **O que observar:**
        - **Burn Rate**: Taxa de consumo de capital por mês
        - **Runway**: Quantos meses a empresa tem antes de ficar sem dinheiro
        - **Status de Caixa**: Indicador de alerta para stress financeiro
        - **Risco de Liquidez**: Classificação do risco baseada no runway
        
        **Para o Santander:** Empresas com runway baixo precisam de crédito urgente. 
        Burn rate alto pode indicar problemas operacionais ou crescimento acelerado.
        """)
    
    if len(dados_analisados_cnpj) > 0:
        # Usar dados pré-calculados da Base_Analisada
        ultimo_mes = dados_analisados_cnpj.iloc[-1]
        
        # Exibir métricas com dados pré-calculados
        col1, col2, col3 = st.columns(3)
        
        with col1:
            burn_rate_mes = ultimo_mes['burn_rate_mes']
            st.metric(
                "Burn Rate Mensal", 
                format_currency(burn_rate_mes),
                help="Taxa média de consumo de capital por mês"
            )
        
        with col2:
            runway_meses = ultimo_mes['runway_meses']            
            # Verificar se runway está vazio, NaN ou infinito
            if pd.isna(runway_meses) or runway_meses == float('inf') or runway_meses > 1000 or runway_meses < 0:
                runway_display = "0"
            else:
                runway_display = f"{runway_meses:.1f}"
            
            st.metric(
                "Runway (Meses)", 
                runway_display,
                help="Meses restantes antes de ficar sem dinheiro"
            )
        
        with col3:
            stress_status = "Alerta" if ultimo_mes['flag_stress_caixa'] == 1 else "Normal"
            st.metric(
                "Status de Caixa", 
                stress_status,
                help="Indicador de stress financeiro"
            )
        
        
    
    elif len(dados_saldo) > 0:
        # Calcular indicadores de liquidez
        saldo_atual = dados_saldo['VL_SLDO'].iloc[-1]
        
        # Burn rate (variação média mensal negativa)
        deltas = dados_saldo['Delta'].dropna()
        burn_rate_mes = abs(deltas[deltas < 0].mean()) if len(deltas[deltas < 0]) > 0 else 0
        
        # Runway
        runway_meses = saldo_atual / burn_rate_mes if burn_rate_mes > 0 else float('inf')
        
        # Flag de stress de caixa
        flag_stress_caixa = (
            saldo_atual < 0 or  # Saldo negativo
            runway_meses < 3 or  # Menos de 3 meses de runway
            burn_rate_mes > saldo_atual * 0.5  # Burn rate muito alto
        )
        
        # Calcular evolução mensal dos indicadores
        dados_evolucao = dados_saldo.copy()
        dados_evolucao['Burn_Rate_Mensal'] = dados_evolucao['Delta'].abs()
        dados_evolucao['Burn_Rate_Mensal'] = dados_evolucao['Burn_Rate_Mensal'].replace(0, np.nan)
        
        # Calcular runway mensal
        dados_evolucao['Runway_Mensal'] = dados_evolucao['VL_SLDO'] / dados_evolucao['Burn_Rate_Mensal']
        dados_evolucao['Runway_Mensal'] = dados_evolucao['Runway_Mensal'].replace([np.inf, -np.inf], np.nan)
        
        # Calcular tendências
        burn_rate_tendencia = dados_evolucao['Burn_Rate_Mensal'].dropna()
        runway_tendencia = dados_evolucao['Runway_Mensal'].dropna()
        
        if len(burn_rate_tendencia) > 1:
            burn_rate_crescimento = burn_rate_tendencia.iloc[-1] - burn_rate_tendencia.iloc[0]
            burn_rate_tendencia_texto = "Aumentando" if burn_rate_crescimento > 0 else "Diminuindo" if burn_rate_crescimento < 0 else "Estável"
        else:
            burn_rate_tendencia_texto = "Estável"
        
        if len(runway_tendencia) > 1:
            runway_crescimento = runway_tendencia.iloc[-1] - runway_tendencia.iloc[0]
            runway_tendencia_texto = "Melhorando" if runway_crescimento > 0 else "Piorando" if runway_crescimento < 0 else "Estável"
        else:
            runway_tendencia_texto = "Estável"
        
        # Exibir métricas com tendências
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Burn Rate Mensal", 
                format_currency(burn_rate_mes),
                delta=f"{burn_rate_tendencia_texto}",
                help="Taxa média de consumo de capital por mês"
            )
        
        with col2:
            if runway_meses == float('inf'):
                runway_display = "∞"
                delta_runway = runway_tendencia_texto
            else:
                runway_display = f"{runway_meses:.1f}"
                delta_runway = runway_tendencia_texto
            
            st.metric(
                "Runway (Meses)", 
                runway_display,
                delta=delta_runway,
                help="Meses restantes antes de ficar sem dinheiro"
            )
        
        with col3:
            stress_status = "ALERTA" if flag_stress_caixa else "NORMAL"
            st.metric(
                "Status de Caixa", 
                stress_status,
                help="Indicador de stress financeiro"
            )
        
        with col4:
            # Indicador visual do risco
            if runway_meses == float('inf'):
                risco_text = "BAIXO"
            elif runway_meses >= 12:
                risco_text = "BAIXO"
            elif runway_meses >= 6:
                risco_text = "MÉDIO"
            elif runway_meses >= 3:
                risco_text = "ALTO"
            else:
                risco_text = "CRÍTICO"
            
            st.metric(
                "Risco de Liquidez", 
                f"{risco_color} {risco_text}",
                help="Classificação do risco baseada no runway"
            )
        
        # Gráfico de evolução dos indicadores
        st.markdown("#### Evolução dos Indicadores de Liquidez")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de Burn Rate mensal
            fig_burn_evolucao = go.Figure()
            
            fig_burn_evolucao.add_trace(go.Scatter(
                x=dados_evolucao['DT_REFE'],
                y=dados_evolucao['Burn_Rate_Mensal'],
                mode='lines+markers',
                name='Burn Rate Mensal',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            
            fig_burn_evolucao.update_layout(
                title="Evolução do Burn Rate",
                xaxis_title="Mês",
                yaxis_title="Burn Rate (R$)",
                height=400
            )
            fig_burn_evolucao.update_yaxes(tickformat='R$ ,.0f')
            
            # Adicionar annotations com valores formatados (apenas se não for NaN)
            for i, (data, valor) in enumerate(zip(dados_evolucao['DT_REFE'], dados_evolucao['Burn_Rate_Mensal'])):
                if not pd.isna(valor):  # Só adicionar se não for NaN
                    fig_burn_evolucao.add_annotation(
                        x=data,
                        y=valor + max(dados_evolucao['Burn_Rate_Mensal'].dropna()) * 0.02,
                        text=f"R$ {valor:,.0f}".replace(",", "."),
                        showarrow=False,
                        font=dict(size=11, color="black"),
                        yanchor='bottom'
                    )
            
            st.plotly_chart(fig_burn_evolucao, use_container_width=True, key="burn_evolucao_chart")
        
        with col2:
            # Gráfico de Runway mensal
            fig_runway_evolucao = go.Figure()
            
            fig_runway_evolucao.add_trace(go.Scatter(
                x=dados_evolucao['DT_REFE'],
                y=dados_evolucao['Runway_Mensal'],
                mode='lines+markers',
                name='Runway Mensal',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
            
            # Adicionar linha de referência para 3 meses
            fig_runway_evolucao.add_hline(y=3, line_dash="dash", line_color="red", 
                                        annotation_text="Limite Crítico (3 meses)")
            
            fig_runway_evolucao.update_layout(
                title="Evolução do Runway",
                xaxis_title="Mês",
                yaxis_title="Runway (Meses)",
                height=400
            )
            
            # Adicionar annotations com valores (apenas se não for NaN ou infinito)
            for i, (data, valor) in enumerate(zip(dados_evolucao['DT_REFE'], dados_evolucao['Runway_Mensal'])):
                if not pd.isna(valor) and valor < 1000:  # Só adicionar se não for NaN ou infinito
                    fig_runway_evolucao.add_annotation(
                        x=data,
                        y=valor + max(dados_evolucao['Runway_Mensal'].dropna()) * 0.02,
                        text=f"{valor:.1f}",
                        showarrow=False,
                        font=dict(size=11, color="black"),
                        yanchor='bottom'
                    )
            
            st.plotly_chart(fig_runway_evolucao, use_container_width=True, key="runway_evolucao_chart")
        
        # Explicação detalhada dos cálculos
        with st.expander("Detalhes dos Cálculos de Liquidez", expanded=False):
            st.markdown("""
            **Metodologia Completa:**
            
            **1. Burn Rate Mensal:**
            - **Cálculo**: Pega todas as variações mensais do saldo (Delta = Saldo_mês_atual - Saldo_mês_anterior)
            - **Filtro**: Considera apenas meses com consumo (Delta < 0)
            - **Fórmula**: Burn Rate = |Média dos Deltas Negativos|
            - **Exemplo**: Se Delta = [-1000, -2000, -1500], então Burn Rate = |-1500| = R$ 1.500/mês
            - **Interpretação**: Quanto a empresa "queima" de capital por mês
            
            **2. Runway (Meses Restantes):**
            - **Fórmula**: Runway = Saldo Atual ÷ Burn Rate Mensal
            - **Exemplo**: Saldo = R$ 15.000, Burn Rate = R$ 1.500 → Runway = 10 meses
            - **Casos especiais**: 
              - Se Burn Rate = 0: Runway = ∞ (empresa não consome capital)
              - Se Saldo < 0: Runway = 0 (já está negativo)
            - **Interpretação**: Tempo de sobrevivência com capital atual
            
            **3. Status de Caixa:**
            - **Critérios de ALERTA** (qualquer um verdadeiro):
              - Saldo atual < 0
              - Runway < 3 meses
              - Burn Rate > 50% do saldo atual
            - **Exemplo**: Saldo = R$ 5.000, Burn Rate = R$ 3.000 → ALERTA (Burn Rate > 50%)
            - **Caso contrário**: NORMAL
            
            **4. Risco de Liquidez:**
            - **Classificação baseada no Runway**:
              - **BAIXO**: Runway ≥ 12 meses (situação confortável)
              - **MÉDIO**: Runway 6-12 meses (atenção necessária)
              - **ALTO**: Runway 3-6 meses (crédito urgente)
              - **CRÍTICO**: Runway < 3 meses (situação crítica)
            
            **5. Tendências:**
            - **Burn Rate**: Compara último mês vs primeiro mês
              - Aumentando: Burn Rate crescendo (piorando)
              - Diminuindo: Burn Rate caindo (melhorando)
            - **Runway**: Compara último mês vs primeiro mês
              - Melhorando: Runway crescendo (mais tempo)
              - Piorando: Runway diminuindo (menos tempo)
            """)
        
    else:
        st.info("Dados insuficientes para calcular indicadores de liquidez.")
    
    # Análise de Tendências (Roll3)
    st.subheader("Análise de Tendências (Roll3)")
    
    # Dropdown informativo para Roll3
    with st.expander("Sobre a Análise Roll3", expanded=False):
        st.markdown("""
        **O que é Roll3:**
        - **Rolling 3-period average**: Média móvel de 3 períodos
        - **Suavização**: Remove ruído e mostra tendências reais
        - **Comparação**: Roll3 vs valores originais mostra estabilidade
        - **Padrões**: Identifica tendências de longo prazo
        
        **Para o Santander:** Roll3 ajuda a identificar se mudanças são temporárias 
        ou parte de uma tendência consistente. Importante para decisões de crédito.
        """)
    
    if len(dados_analisados_cnpj) > 0:
        # Usar dados pré-calculados da Base_Analisada
        ultimo_mes = dados_analisados_cnpj.iloc[-1]
        
        # Exibir métricas Roll3 pré-calculadas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            roll3_volume = ultimo_mes['roll3_vol_volume_total_mes']
            st.metric("Roll3 Volume Total", format_currency(roll3_volume))
        
        with col2:
            grow_volume = ultimo_mes['grow_volume_total_mes']
            st.metric("Crescimento Volume", f"{grow_volume:.2%}")
        
        with col3:
            volatilidade = ultimo_mes['q75_volat']
            st.metric("Volatilidade", f"{volatilidade:.2f}")
        
        with col4:
            # Comparar com percentil 75 da volatilidade
            if volatilidade >= ultimo_mes['q75_volat']:
                volat_status = "Alta"
            elif volatilidade >= ultimo_mes['q75_volat'] * 0.5:
                volat_status = "Média"
            else:
                volat_status = "Baixa"
            st.metric("Status Volatilidade", volat_status)
        
        # Gráfico de evolução do volume com Roll3
        if len(dados_analisados_cnpj) > 1:
            dados_analisados_sorted = dados_analisados_cnpj.sort_values('MES')
            
            fig_volume_roll3 = go.Figure()
            
            fig_volume_roll3.add_trace(go.Scatter(
                x=dados_analisados_sorted['MES'],
                y=dados_analisados_sorted['volume_total_mes'],
                mode='lines+markers',
                name='Volume Original',
                line=dict(color='blue', width=2)
            ))
            
            fig_volume_roll3.add_trace(go.Scatter(
                x=dados_analisados_sorted['MES'],
                y=dados_analisados_sorted['roll3_vol_volume_total_mes'],
                mode='lines+markers',
                name='Volume Roll3',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            fig_volume_roll3.update_layout(
                title="Volume Total vs Roll3 (Média Móvel)",
                xaxis_title="Mês",
                yaxis_title="Volume (R$)",
                height=400
            )
            fig_volume_roll3.update_yaxes(tickformat='R$ ,.0f')
            
            # Adicionar annotations com valores formatados (apenas se não for NaN)
            for i, (mes, volume_orig, volume_roll3) in enumerate(zip(dados_analisados_sorted['MES'], 
                                                                    dados_analisados_sorted['volume_total_mes'],
                                                                    dados_analisados_sorted['roll3_vol_volume_total_mes'])):
                if not pd.isna(volume_orig) and not pd.isna(volume_roll3):
                    fig_volume_roll3.add_annotation(
                        x=mes,
                        y=max(volume_orig, volume_roll3) + max(dados_analisados_sorted['volume_total_mes']) * 0.02,
                        text=f"O:{volume_orig:,.0f} R:{volume_roll3:,.0f}".replace(",", "."),
                        showarrow=False,
                        font=dict(size=10, color="black"),
                        yanchor='bottom'
                    )
            
            st.plotly_chart(fig_volume_roll3, use_container_width=True)
    
    elif len(dados_saldo) > 0:
        # Calcular Roll3 para saldo
        dados_saldo_sorted = dados_saldo.sort_values('DT_REFE')
        dados_saldo_sorted['Saldo_Roll3'] = dados_saldo_sorted['VL_SLDO'].rolling(window=3, min_periods=1).mean()
        
        # Calcular Roll3 para burn rate
        dados_saldo_sorted['Burn_Rate_Roll3'] = dados_saldo_sorted['Delta'].abs().rolling(window=3, min_periods=1).mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico Saldo vs Roll3
            fig_saldo_roll3 = go.Figure()
            
            fig_saldo_roll3.add_trace(go.Scatter(
                x=dados_saldo_sorted['DT_REFE'],
                y=dados_saldo_sorted['VL_SLDO'],
                mode='lines+markers',
                name='Saldo Original',
                line=dict(color='blue', width=2)
            ))
            
            fig_saldo_roll3.add_trace(go.Scatter(
                x=dados_saldo_sorted['DT_REFE'],
                y=dados_saldo_sorted['Saldo_Roll3'],
                mode='lines+markers',
                name='Saldo Roll3',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            fig_saldo_roll3.update_layout(
                title="Saldo vs Roll3 (Média Móvel)",
                xaxis_title="Data",
                yaxis_title="Saldo (R$)",
                height=400
            )
            fig_saldo_roll3.update_yaxes(tickformat='R$ ,.0f')
            
            # Adicionar annotations com valores formatados (apenas se não for NaN)
            for i, (data, saldo_orig, saldo_roll3) in enumerate(zip(dados_saldo_sorted['DT_REFE'], 
                                                                   dados_saldo_sorted['VL_SLDO'],
                                                                   dados_saldo_sorted['Saldo_Roll3'])):
                if not pd.isna(saldo_orig) and not pd.isna(saldo_roll3):
                    fig_saldo_roll3.add_annotation(
                        x=data,
                        y=max(saldo_orig, saldo_roll3) + max(dados_saldo_sorted['VL_SLDO']) * 0.02,
                        text=f"O:{saldo_orig:,.0f} R:{saldo_roll3:,.0f}".replace(",", "."),
                        showarrow=False,
                        font=dict(size=10, color="black"),
                        yanchor='bottom'
                    )
            
            st.plotly_chart(fig_saldo_roll3, use_container_width=True)
        
        with col2:
            # Gráfico Burn Rate vs Roll3
            fig_burn_roll3 = go.Figure()
            
            fig_burn_roll3.add_trace(go.Scatter(
                x=dados_saldo_sorted['DT_REFE'],
                y=dados_saldo_sorted['Delta'].abs(),
                mode='lines+markers',
                name='Burn Rate Original',
                line=dict(color='orange', width=2)
            ))
            
            fig_burn_roll3.add_trace(go.Scatter(
                x=dados_saldo_sorted['DT_REFE'],
                y=dados_saldo_sorted['Burn_Rate_Roll3'],
                mode='lines+markers',
                name='Burn Rate Roll3',
                line=dict(color='red', width=3, dash='dash')
            ))
            
            fig_burn_roll3.update_layout(
                title="Burn Rate vs Roll3 (Média Móvel)",
                xaxis_title="Data",
                yaxis_title="Burn Rate (R$)",
                height=400
            )
            fig_burn_roll3.update_yaxes(tickformat='R$ ,.0f')
            
            # Adicionar annotations com valores formatados (apenas se não for NaN)
            for i, (data, burn_orig, burn_roll3) in enumerate(zip(dados_saldo_sorted['DT_REFE'], 
                                                                  dados_saldo_sorted['Delta'].abs(),
                                                                  dados_saldo_sorted['Burn_Rate_Roll3'])):
                if not pd.isna(burn_roll3):  # Só adicionar se não for NaN
                    fig_burn_roll3.add_annotation(
                        x=data,
                        y=max(burn_orig, burn_roll3) + max(dados_saldo_sorted['Burn_Rate_Roll3'].dropna()) * 0.02,
                        text=f"O:{burn_orig:,.0f} R:{burn_roll3:,.0f}".replace(",", "."),
                        showarrow=False,
                        font=dict(size=10, color="black"),
                        yanchor='bottom'
                    )
            
            st.plotly_chart(fig_burn_roll3, use_container_width=True)
        
        # Métricas Roll3
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            saldo_roll3_atual = dados_saldo_sorted['Saldo_Roll3'].iloc[-1]
            st.metric("Saldo Roll3 Atual", format_currency(saldo_roll3_atual))
        
        with col2:
            burn_rate_roll3_atual = dados_saldo_sorted['Burn_Rate_Roll3'].iloc[-1]
            st.metric("Burn Rate Roll3", format_currency(burn_rate_roll3_atual))
        
        with col3:
            variacao_roll3 = dados_saldo_sorted['Saldo_Roll3'].iloc[-1] - dados_saldo_sorted['Saldo_Roll3'].iloc[0]
            st.metric("Variação Roll3", format_currency(variacao_roll3))
        
        with col4:
            estabilidade = 1 - (dados_saldo_sorted['Saldo_Roll3'].std() / dados_saldo_sorted['Saldo_Roll3'].mean()) if dados_saldo_sorted['Saldo_Roll3'].mean() != 0 else 0
            st.metric("Estabilidade Roll3", f"{estabilidade:.2f}")
    

if __name__ == "__main__":
    main()
