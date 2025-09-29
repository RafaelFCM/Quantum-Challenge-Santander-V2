<div align="center">
  
# 📹 LINK DA APRESENTAÇÃO: https://youtu.be/38opEqyDM8w

</div>

# 🏦 Dashboard Santander - Análise de CNPJs

Dashboard interativo para análise de dados de CNPJs clientes do Santander, desenvolvido em Python com Streamlit.

## 🔄 Fluxo do Processo

📊 **Fluxograma Interativo**: Abra o arquivo `fluxograma_santander.html` no navegador para visualizar o processo completo com animações e detalhes.

**Resumo do Fluxo:**

1. **📁 Dados Brutos** → 150k registros em CSV
2. **🔄 Processamento** → Limpeza e validação
3. **📊 Análise** → Estatísticas e padrões
4. **🧮 Métricas** → 8 métricas avançadas calculadas
5. **📈 Visualizações** → Dashboards interativos
6. **🤖 Sistema RAG** → IA para análises inteligentes
7. **💡 Insights** → Decisões para o Santander

## 🎯 Principais Funcionalidades

### 📊 Métricas Avançadas

- **HHI (Herfindahl-Hirschman Index)**: Mede concentração de relacionamentos
- **Burn Rate**: Taxa de consumo mensal de capital
- **Runway**: Meses restantes antes de ficar sem dinheiro
- **Score de Saúde**: Avaliação 0-100 da saúde financeira
- **Momento de Vida**: Iniciante/Desenvolvimento/Madura/Declínio

### 🤖 Sistema RAG Inteligente

- **ChatGPT-4**: Análises avançadas em linguagem natural
- **Contexto Dinâmico**: Dados específicos por CNPJ
- **Respostas Detalhadas**: CNPJs específicos com métricas exatas

### 📈 Visualizações Interativas

- **Dashboard Geral**: Visão macro da carteira (10k CNPJs)
- **Análise Individual**: Deep dive por CNPJ específico
- **Rede de Relacionamentos**: Mapa de conexões comerciais
- **Análise Temporal**: Padrões mensais de atividade

## 🏆 Resultados para o Santander

### 📊 Métricas Principais

- **10.000 CNPJs** analisados
- **100.000 transações** processadas
- **R$ 5.693.144.404** em volume total
- **5 meses** de dados históricos (Jan-Mai 2025)

### 🎯 Capacidades Desenvolvidas

1. **Identificação Automática** de CNPJs ideais para crédito
2. **Detecção de Riscos** com alertas específicos
3. **Análise de Setores** para diversificação da carteira
4. **Recomendações Personalizadas** baseadas em dados
5. **Interface Intuitiva** para análise em linguagem natural

### 💡 Exemplos de Insights Gerados

- **Top CNPJs para Crédito**: Ranqueados por score composto
- **Empresas em Risco**: Identificadas por critérios específicos
- **Setores Promissores**: Baseados em volume e estabilidade
- **Oportunidades de Negócio**: Produtos adequados por perfil

## 🚀 Como Usar

### 1. **Instalação**

```bash
# Clone o repositório
git clone [url-do-repositorio]
cd Quantum-Cursor

# Instale as dependências
pip install -r requirements.txt
```

### 2. **Configuração da API (Opcional)**

Para usar o Assistente IA, configure a chave da OpenAI:

```bash
export OPENAI_API_KEY="sua_chave_aqui"
```

### 3. **Execução**

```bash
streamlit run dashboard_santander.py
```

## 📋 Funcionalidades

### 🏠 Dashboard Geral

- **KPIs Principais**: Total de CNPJs, transações, volume e faturamento médio
- **Análise por Setor**: Distribuição de empresas por CNAE
- **Distribuição de Faturamento**: Histograma de valores
- **Evolução Temporal**: Transações e volume por mês
- **Tipos de Transação**: Distribuição PIX, TED, BOLETO, SISTEMICO
- **Visualizações Avançadas**:
  - 🌊 Gráfico Sankey para fluxo de transações
  - 📅 Heatmap temporal
  - ⚠️ Análise de risco por score
  - 🏭 Análise por setor

### 🔍 Análise Individual

- **Busca por CNPJ**: Seleção de empresa específica
- **Informações da Empresa**: Setor, faturamento, saldo, idade
- **Análise de Transações**: Total pago/recebido
- **Fluxo de Caixa Temporal**: Gráfico de dispersão temporal
- **Parceiros Comerciais**: Principais destinos e fontes
- **Rede de Relacionamentos**: Visualização em grafo interativo

### 📊 Relatórios Avançados

- **Análise de Risco por Setor**: Setores com maior risco
- **Matriz de Correlação**: Correlações entre variáveis
- **Top Empresas**: Ranking por faturamento
- **Padrões de Transação**: Análise por hora do dia

## 🔧 Arquitetura Técnica

### **Frontend**

- **Streamlit**: Interface web responsiva
- **Plotly**: Visualizações interativas
- **CSS Personalizado**: Design profissional Santander

### **Backend**

- **Python**: Processamento de dados
- **Pandas**: Manipulação de DataFrames
- **NumPy**: Cálculos matemáticos
- **Scikit-learn**: Análise estatística

### **IA e Análise**

- **OpenAI GPT-4**: Sistema RAG
- **NetworkX**: Análise de redes
- **Seaborn/Matplotlib**: Visualizações estatísticas

## 📁 Estrutura do Projeto

```
Quantum-Cursor/
├── dados/
│   ├── Bases_Legenda.csv
│   ├── Base_Infos.csv
│   └── Base_Transacoes.csv
├── dashboard_santander.py
├── visualizacoes_avancadas.py
├── rag_system.py
├── fluxograma_santander.html
├── requirements.txt
├── README.md
└── RAG_INSTRUCOES.md
```

## 📊 Dados

O dashboard utiliza três arquivos CSV:

- **Bases_Legenda.csv**: Definições das variáveis
- **Base_Infos.csv**: Informações de 10.000 CNPJs (5 meses)
- **Base_Transacoes.csv**: 100.000 transações entre CNPJs (3 meses)

## 📈 Impacto Esperado

### **Para Analistas de Crédito**

- ⚡ **Decisões 10x mais rápidas** com dados estruturados
- 🎯 **Maior precisão** na avaliação de riscos
- 📊 **Visão completa** do perfil do cliente

### **Para Gestores**

- 📈 **Insights estratégicos** sobre a carteira
- 🎯 **Identificação de oportunidades** de negócio
- ⚠️ **Antecipação de riscos** sistêmicos

### **Para o Banco**

- 💰 **Redução de inadimplência** com análise preditiva
- 🚀 **Aumento de receita** com produtos adequados
- 🏆 **Vantagem competitiva** com tecnologia avançada

## 🚀 Próximos Passos

1. **Integração com Sistemas** internos do Santander
2. **Expansão de Dados** para mais períodos históricos
3. **Machine Learning** para predição de inadimplência
4. **API REST** para integração com outros sistemas
5. **Dashboard Mobile** para acesso em campo

# 🤖 Sistema RAG - Assistente de Análise Santander

## 📋 Configuração da API OpenAI

Para usar o **Assistente IA**, você precisa configurar a chave da API do OpenAI:

### 🔑 Como obter a chave:

1. **Acesse**: https://platform.openai.com/api-keys
2. **Faça login** na sua conta OpenAI
3. **Clique em "Create new secret key"**
4. **Copie a chave** gerada

### ⚙️ Como configurar:

#### **Windows (PowerShell):**

$env:OPENAI_API_KEY="sua_chave_aqui"

### 🚀 Como usar:

1. **Configure a variável de ambiente** com sua chave, no terminal vs code
2. **Execute o dashboard**: `streamlit run dashboard_santander.py`

## 💡 Exemplos de Perguntas:

### 📊 Análise Geral:

- "Quais setores têm maior volume de transações?"
- "Qual é o perfil médio dos CNPJs da carteira?"
- "Quais empresas são mais ativas comercialmente?"

### 🔍 Análise Individual:

- "Qual CNPJ tem maior risco de liquidez?"
- "Quais empresas estão em crescimento?"
- "Quem tem relacionamentos mais diversificados?"

### ⚠️ Identificação de Riscos:

- "Identifique CNPJs com problemas de liquidez"
- "Quais empresas têm alta concentração de relacionamentos?"
- "Mostre empresas com saldo em declínio"

### 💡 Oportunidades:

- "Quais setores são mais seguros para crédito?"
- "Identifique empresas com potencial de crescimento"
- "Quais CNPJs são candidatos a produtos premium?"

## 🧠 Como Funciona:

### **RAG (Retrieval-Augmented Generation):**

1. **Retrieval**: Busca dados relevantes dos CNPJs
2. **Augmentation**: Enriquece o contexto com métricas calculadas
3. **Generation**: ChatGPT-4 gera resposta inteligente

---

**Desenvolvido para o Santander** 🏦
