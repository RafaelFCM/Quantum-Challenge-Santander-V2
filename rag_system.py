import streamlit as st
import pandas as pd
import json
import requests
from typing import Dict, List, Any
import os
from datetime import datetime

class SantanderRAG:
    """Sistema RAG para análise de CNPJs do Santander"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.conversation_history = []
        
    def prepare_context(self, df_infos: pd.DataFrame, df_transacoes: pd.DataFrame, cnpj_selecionado: str = None) -> str:
        """Preparar contexto estruturado dos dados para o RAG"""
        
        context_parts = []
        
        # Contexto geral
        total_cnpjs = df_infos['ID'].nunique()
        total_transacoes = len(df_transacoes)
        volume_total = df_transacoes['VL'].sum()
        faturamento_medio = df_infos.drop_duplicates(subset=['ID'])['VL_FATU'].mean()
        
        context_parts.append(f"""
CONTEXTO GERAL DO SANTANDER:
- Total de CNPJs na carteira: {total_cnpjs:,}
- Total de transações: {total_transacoes:,}
- Volume total movimentado: R$ {volume_total:,.2f}
- Faturamento médio por CNPJ: R$ {faturamento_medio:,.2f}
- Período dos dados: Janeiro a Maio de 2025
""")
        
        # Top setores com detalhes
        df_unique_setores = df_infos.drop_duplicates(subset=['ID'])
        top_setores = df_unique_setores['DS_CNAE'].value_counts().head(10)
        context_parts.append(f"""
TOP 10 SETORES POR QUANTIDADE DE CNPJs:
{top_setores.to_string()}
""")
        
        # Top empresas por faturamento com detalhes completos
        df_unique_fatu = df_infos.drop_duplicates(subset=['ID'])
        top_empresas = df_unique_fatu.nlargest(20, 'VL_FATU')[['ID', 'VL_FATU', 'DS_CNAE']].copy()
        top_empresas['VL_FATU'] = top_empresas['VL_FATU'].apply(lambda x: f"R$ {x:,.2f}")
        context_parts.append(f"""
TOP 20 EMPRESAS POR FATURAMENTO (com setor):
{top_empresas.to_string()}
""")
        
        # Análise de saúde financeira dos top CNPJs usando as mesmas funções do dashboard
        context_parts.append("""
ANÁLISE DE SAÚDE FINANCEIRA DOS TOP CNPJs:
""")
        
        # Importar as funções de cálculo do dashboard
        from dashboard_santander import calcular_saude_empresa, calcular_risco_dependencia, calcular_score_santander
        
        for _, empresa in df_unique_fatu.nlargest(10, 'VL_FATU').iterrows():
            cnpj_id = empresa['ID']
            dados_cnpj = df_infos[df_infos['ID'] == cnpj_id].copy()
            dados_cnpj = dados_cnpj.sort_values('DT_REFE')
            
            if len(dados_cnpj) > 0:
                saldo_atual = dados_cnpj['VL_SLDO'].iloc[-1]
                saldo_inicial = dados_cnpj['VL_SLDO'].iloc[0]
                variacao_saldo = saldo_atual - saldo_inicial
                crescimento_percentual = (variacao_saldo / abs(saldo_inicial)) * 100 if saldo_inicial != 0 else 0
                
                # Calcular estabilidade
                cv_saldo = dados_cnpj['VL_SLDO'].std() / dados_cnpj['VL_SLDO'].mean() if dados_cnpj['VL_SLDO'].mean() != 0 else 1
                
                # Calcular relacionamentos
                transacoes_cnpj = df_transacoes[
                    (df_transacoes['ID_PGTO'] == cnpj_id) | 
                    (df_transacoes['ID_RCBE'] == cnpj_id)
                ]
                num_relacionamentos = len(set(transacoes_cnpj['ID_PGTO'].unique()) | set(transacoes_cnpj['ID_RCBE'].unique()))
                volume_transacoes = transacoes_cnpj['VL'].sum()
                
                # Calcular HHI
                pagamentos = transacoes_cnpj[transacoes_cnpj['ID_PGTO'] == cnpj_id]
                recebimentos = transacoes_cnpj[transacoes_cnpj['ID_RCBE'] == cnpj_id]
                
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
                
                # Calcular burn rate
                deltas = dados_cnpj['VL_SLDO'].diff().dropna()
                burn_rate = abs(deltas[deltas < 0].mean()) if len(deltas[deltas < 0]) > 0 else 0
                runway = saldo_atual / burn_rate if burn_rate > 0 else float('inf')
                
                # Usar as mesmas funções do dashboard para métricas avançadas
                score_saude, categoria_saude = calcular_saude_empresa(df_infos, df_transacoes, cnpj_id)
                score_risco, categoria_risco = calcular_risco_dependencia(df_transacoes, cnpj_id)
                score_santander, detalhes_santander = calcular_score_santander(df_infos, df_transacoes, cnpj_id)
                
                context_parts.append(f"""
CNPJ_{cnpj_id}:
- Faturamento: R$ {empresa['VL_FATU']:,.2f}
- Setor: {empresa['DS_CNAE']}
- Saldo Atual: R$ {saldo_atual:,.2f}
- Variação do Saldo: R$ {variacao_saldo:,.2f} ({crescimento_percentual:+.1f}%)
- Estabilidade (CV): {cv_saldo:.3f}
- Número de Relacionamentos: {num_relacionamentos}
- Volume de Transações: R$ {volume_transacoes:,.2f}
- HHI Médio (Concentração): {hhi_medio:.3f}
- Burn Rate: R$ {burn_rate:,.2f}/mês
- Runway: {runway:.1f} meses
- Score de Saúde: {score_saude:.1f} ({categoria_saude})
- Risco de Dependência: {score_risco:.1f} ({categoria_risco})
- Score Santander: {score_santander:.1f}
""")
        
        # Análise de risco por setor
        context_parts.append("""
ANÁLISE DE RISCO POR SETOR:
""")
        
        for setor in df_unique_setores['DS_CNAE'].value_counts().head(5).index:
            empresas_setor = df_unique_setores[df_unique_setores['DS_CNAE'] == setor]
            faturamento_medio_setor = empresas_setor['VL_FATU'].mean()
            num_empresas_setor = len(empresas_setor)
            
            # Calcular volume de transações do setor
            transacoes_setor = df_transacoes[
                (df_transacoes['ID_PGTO'].isin(empresas_setor['ID'])) |
                (df_transacoes['ID_RCBE'].isin(empresas_setor['ID']))
            ]
            volume_setor = transacoes_setor['VL'].sum()
            
            context_parts.append(f"""
{setor}:
- Número de empresas: {num_empresas_setor}
- Faturamento médio: R$ {faturamento_medio_setor:,.2f}
- Volume de transações: R$ {volume_setor:,.2f}
""")
        
        # Análise de empresas com melhor perfil para crédito usando Score Santander
        context_parts.append("""
RANKING DE EMPRESAS PARA CRÉDITO (baseado no Score Santander):
""")
        
        # Calcular Score Santander para todos os CNPJs
        df_ranking = df_unique_fatu.copy()
        df_ranking['Score_Santander'] = 0
        df_ranking['Score_Saude'] = 0
        df_ranking['Risco_Dependencia'] = 0
        
        for idx, empresa in df_ranking.iterrows():
            cnpj_id = empresa['ID']
            
            # Usar as mesmas funções do dashboard
            score_saude, categoria_saude = calcular_saude_empresa(df_infos, df_transacoes, cnpj_id)
            score_risco, categoria_risco = calcular_risco_dependencia(df_transacoes, cnpj_id)
            score_santander, detalhes_santander = calcular_score_santander(df_infos, df_transacoes, cnpj_id)
            
            df_ranking.loc[idx, 'Score_Santander'] = score_santander
            df_ranking.loc[idx, 'Score_Saude'] = score_saude
            df_ranking.loc[idx, 'Risco_Dependencia'] = score_risco
        
        # Top 15 empresas para crédito baseado no Score Santander
        top_credito = df_ranking.nlargest(15, 'Score_Santander')[['ID', 'VL_FATU', 'DS_CNAE', 'Score_Santander', 'Score_Saude', 'Risco_Dependencia']]
        
        context_parts.append(f"""
TOP 15 EMPRESAS PARA CRÉDITO (Score Santander):
{top_credito.to_string()}
""")
        
        # Análise de empresas em risco
        context_parts.append("""
EMPRESAS COM ALERTAS DE RISCO:
""")
        
        empresas_risco = []
        for _, empresa in df_unique_fatu.iterrows():
            cnpj_id = empresa['ID']
            dados_cnpj = df_infos[df_infos['ID'] == cnpj_id].copy()
            dados_cnpj = dados_cnpj.sort_values('DT_REFE')
            
            if len(dados_cnpj) > 0:
                saldo_atual = dados_cnpj['VL_SLDO'].iloc[-1]
                saldo_inicial = dados_cnpj['VL_SLDO'].iloc[0]
                variacao_saldo = saldo_atual - saldo_inicial
                
                # Calcular burn rate
                deltas = dados_cnpj['VL_SLDO'].diff().dropna()
                burn_rate = abs(deltas[deltas < 0].mean()) if len(deltas[deltas < 0]) > 0 else 0
                runway = saldo_atual / burn_rate if burn_rate > 0 else float('inf')
                
                # Critérios de risco
                if (saldo_atual < 0 or 
                    runway < 3 or 
                    variacao_saldo < -empresa['VL_FATU'] * 0.1):
                    
                    empresas_risco.append({
                        'CNPJ': cnpj_id,
                        'Faturamento': empresa['VL_FATU'],
                        'Saldo_Atual': saldo_atual,
                        'Variacao': variacao_saldo,
                        'Runway': runway,
                        'Setor': empresa['DS_CNAE']
                    })
        
        if empresas_risco:
            for empresa in empresas_risco[:10]:  # Top 10 em risco
                context_parts.append(f"""
CNPJ_{empresa['CNPJ']}:
- Faturamento: R$ {empresa['Faturamento']:,.2f}
- Saldo Atual: R$ {empresa['Saldo_Atual']:,.2f}
- Variação: R$ {empresa['Variacao']:,.2f}
- Runway: {empresa['Runway']:.1f} meses
- Setor: {empresa['Setor']}
- ALERTA: {'Saldo negativo' if empresa['Saldo_Atual'] < 0 else 'Runway baixo' if empresa['Runway'] < 3 else 'Declínio significativo'}
""")
        
        # Análise específica do CNPJ se selecionado
        if cnpj_selecionado:
            dados_cnpj = df_infos[df_infos['ID'] == cnpj_selecionado].copy()
            transacoes_cnpj = df_transacoes[
                (df_transacoes['ID_PGTO'] == cnpj_selecionado) | 
                (df_transacoes['ID_RCBE'] == cnpj_selecionado)
            ].copy()
            
            if len(dados_cnpj) > 0:
                saldo_atual = dados_cnpj['VL_SLDO'].iloc[-1]
                faturamento = dados_cnpj['VL_FATU'].iloc[0]
                setor = dados_cnpj['DS_CNAE'].iloc[0]
                idade_meses = len(dados_cnpj)
                
                # Relacionamentos
                pagamentos = transacoes_cnpj[transacoes_cnpj['ID_PGTO'] == cnpj_selecionado]
                recebimentos = transacoes_cnpj[transacoes_cnpj['ID_RCBE'] == cnpj_selecionado]
                
                num_pagamentos = len(pagamentos['ID_RCBE'].unique()) if len(pagamentos) > 0 else 0
                num_recebimentos = len(recebimentos['ID_PGTO'].unique()) if len(recebimentos) > 0 else 0
                volume_pagamentos = pagamentos['VL'].sum() if len(pagamentos) > 0 else 0
                volume_recebimentos = recebimentos['VL'].sum() if len(recebimentos) > 0 else 0
                
                # Evolução do saldo
                dados_cnpj_sorted = dados_cnpj.sort_values('DT_REFE')
                saldo_inicial = dados_cnpj_sorted['VL_SLDO'].iloc[0]
                variacao_total = saldo_atual - saldo_inicial
                
                # Calcular métricas avançadas usando as mesmas funções do dashboard
                score_saude, categoria_saude = calcular_saude_empresa(df_infos, df_transacoes, cnpj_selecionado)
                score_risco, categoria_risco = calcular_risco_dependencia(df_transacoes, cnpj_selecionado)
                score_santander, detalhes_santander = calcular_score_santander(df_infos, df_transacoes, cnpj_selecionado)
                
                # Calcular burn rate e runway
                deltas = dados_cnpj['VL_SLDO'].diff().dropna()
                burn_rate = abs(deltas[deltas < 0].mean()) if len(deltas[deltas < 0]) > 0 else 0
                runway = saldo_atual / burn_rate if burn_rate > 0 else float('inf')
                
                # Calcular HHI
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
                
                context_parts.append(f"""
ANÁLISE ESPECÍFICA DO CNPJ {cnpj_selecionado}:
- Saldo atual: R$ {saldo_atual:,.2f}
- Faturamento: R$ {faturamento:,.2f}
- Setor: {setor}
- Idade (meses): {idade_meses}
- Variação do saldo: R$ {variacao_total:,.2f}
- Parceiros de pagamento: {num_pagamentos}
- Parceiros de recebimento: {num_recebimentos}
- Volume de pagamentos: R$ {volume_pagamentos:,.2f}
- Volume de recebimentos: R$ {volume_recebimentos:,.2f}
- HHI Médio: {hhi_medio:.3f}
- Burn Rate: R$ {burn_rate:,.2f}/mês
- Runway: {runway:.1f} meses
- Score de Saúde: {score_saude:.1f} ({categoria_saude})
- Risco de Dependência: {score_risco:.1f} ({categoria_risco})
- Score Santander: {score_santander:.1f}

EVOLUÇÃO MENSAL DO SALDO:
{dados_cnpj_sorted[['DT_REFE', 'VL_SLDO']].to_string()}
""")
                
                # Top relacionamentos
                if len(pagamentos) > 0:
                    top_pagamentos = pagamentos.groupby('ID_RCBE')['VL'].sum().sort_values(ascending=False).head(3)
                    context_parts.append(f"""
TOP 3 RELACIONAMENTOS DE PAGAMENTO:
{top_pagamentos.to_string()}
""")
                
                if len(recebimentos) > 0:
                    top_recebimentos = recebimentos.groupby('ID_PGTO')['VL'].sum().sort_values(ascending=False).head(3)
                    context_parts.append(f"""
TOP 3 RELACIONAMENTOS DE RECEBIMENTO:
{top_recebimentos.to_string()}
""")
        
        return "\n".join(context_parts)
    
    def generate_response(self, question: str, context: str) -> str:
        """Gerar resposta usando ChatGPT com contexto"""
        
        if not self.api_key:
            return "Erro: Chave da API OpenAI não configurada. Configure a variável OPENAI_API_KEY."
        
        # Preparar mensagens para o ChatGPT
        system_prompt = """Você é um analista financeiro especializado do Banco Santander. 
        Sua função é analisar dados de CNPJs e fornecer insights valiosos para decisões de crédito e investimento.
        
        REGRAS CRÍTICAS:
        1. Sempre responda em português brasileiro
        2. USE APENAS os dados fornecidos no contexto - NÃO invente números
        3. SEMPRE cite números específicos, CNPJs exatos e métricas calculadas do contexto
        4. Seja detalhado e específico, não genérico
        5. Cite CNPJs pelo ID exato (ex: CNPJ_08809, CNPJ_07238)
        6. Use valores monetários formatados (R$ 1.000.000,00)
        7. Cite métricas específicas: HHI, Burn Rate, Runway, CV, Score de Saúde, Risco de Dependência, Score Santander
        8. Foque em insights práticos para o Santander
        9. Sugira ações concretas quando apropriado
        10. Use formatação markdown para melhor legibilidade
        
        CONTEXTO DOS DADOS:
        - Período: Janeiro a Maio de 2025
        - Dados incluem: saldos, faturamentos, transações, relacionamentos comerciais
        - Métricas calculadas: HHI (concentração), Burn Rate (consumo mensal), Runway (meses restantes), CV (estabilidade)
        - Variáveis avançadas: Saúde da Empresa (0-100), Risco de Dependência (0-100), Score Santander (0-100)
        
        IMPORTANTE: Use EXATAMENTE os valores fornecidos no contexto. Se uma métrica não estiver disponível no contexto, diga "N/D" ao invés de inventar valores.
        
        EXEMPLO DE RESPOSTA DETALHADA:
        "Os 3 melhores CNPJs para crédito são:
        
        1. **CNPJ_08809**: 
           - Faturamento: R$ 199.929.313,00
           - Saldo atual: R$ 2.500.000,00
           - Score de Saúde: 85,2 (Alta)
           - Risco de Dependência: 25,3 (Baixo)
           - Score Santander: 78,5
           - HHI: 0,234 (baixa concentração)
           - Runway: 18,5 meses
           - Setor: Comércio de Equipamentos
        
        2. **CNPJ_07238**: 
           - Faturamento: R$ 199.496.386,00
           - Saldo atual: R$ 1.800.000,00
           - Score de Saúde: 72,1 (Média)
           - Risco de Dependência: 45,2 (Médio)
           - Score Santander: 65,8
           - HHI: 0,312 (concentração moderada)
           - Runway: 12,3 meses
           - Setor: Serviços Financeiros"
        """
        
        user_prompt = f"""
CONTEXTO COMPLETO DOS DADOS:
{context}

PERGUNTA DO USUÁRIO:
{question}

INSTRUÇÕES CRÍTICAS:
- Use APENAS os dados fornecidos no contexto acima
- NÃO invente números ou métricas que não estão no contexto
- Cite CNPJs específicos com seus IDs exatos do contexto
- Use valores monetários formatados (R$ 1.000.000,00)
- Cite métricas calculadas: HHI, Burn Rate, Runway, CV, Score de Saúde, Risco de Dependência, Score Santander
- Se uma métrica não estiver no contexto, diga "N/D" ao invés de inventar
- Seja detalhado e específico, não genérico
- Forneça insights práticos para decisões do Santander
- Use formatação markdown para organizar a resposta

IMPORTANTE: Verifique se todos os números que você citar estão EXATAMENTE no contexto fornecido acima.

Responda de forma completa e detalhada, citando números específicos e CNPJs exatos do contexto.
"""
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Erro na API: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Erro ao conectar com a API: {str(e)}"
    
    def add_to_history(self, question: str, answer: str):
        """Adicionar conversa ao histórico"""
        self.conversation_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "question": question,
            "answer": answer
        })
        
        # Manter apenas as últimas 10 conversas
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_history(self) -> List[Dict]:
        """Obter histórico de conversas"""
        return self.conversation_history

def create_chat_interface():
    """Criar interface de chat no Streamlit"""
    
    st.subheader("Assistente de Análise Santander")
    
    # Inicializar RAG se não existir
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = SantanderRAG()
    
    # Verificar se API key está configurada
    if not os.getenv('OPENAI_API_KEY'):
        st.warning("Para usar o assistente, configure a variável de ambiente OPENAI_API_KEY")
        st.info("Exemplo: `export OPENAI_API_KEY=sua_chave_aqui`")
        return
    
    # Input para pergunta
    question = st.text_input(
        "Faça uma pergunta sobre os CNPJs ou análises:",
        placeholder="Ex: Qual CNPJ tem maior risco de liquidez? Quais setores são mais seguros para crédito?",
        key="chat_input"
    )
    
    # Botão para enviar
    send_button = st.button("Enviar", type="primary")
    
    # Processar pergunta
    if send_button and question:
        with st.spinner("Analisando dados e gerando resposta..."):
            # Preparar contexto baseado no CNPJ selecionado (se houver)
            cnpj_selecionado = st.session_state.get('cnpj_selecionado', None)
            
            # Obter dados do contexto do Streamlit
            df_infos = st.session_state.get('df_infos')
            df_transacoes = st.session_state.get('df_transacoes')
            
            if df_infos is not None and df_transacoes is not None:
                context = st.session_state.rag_system.prepare_context(
                    df_infos, df_transacoes, cnpj_selecionado
                )
                
                # Gerar resposta
                answer = st.session_state.rag_system.generate_response(question, context)
                
                # Adicionar ao histórico
                st.session_state.rag_system.add_to_history(question, answer)
                
                # Exibir resposta
                st.markdown("### Resposta do Assistente:")
                st.markdown(answer)
            else:
                st.error("Dados não disponíveis. Carregue os dados primeiro.")
    
    # Exibir histórico de conversas
    history = st.session_state.rag_system.get_conversation_history()
    
    if history:
        st.markdown("---")
        st.subheader("Histórico de Conversas")
        
        for conv in reversed(history[-5:]):  # Mostrar últimas 5 conversas
            with st.expander(f"{conv['timestamp']} - {conv['question'][:50]}..."):
                st.markdown(f"**Pergunta:** {conv['question']}")
                st.markdown(f"**Resposta:** {conv['answer']}")
    
