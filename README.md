# 🤟 S.I.N.A.I.S - Sistema Integrado de Tradução e Processamento de Sinais

Este é o repositório principal do **S.I.N.A.I.S**, um tradutor de LIBRAS em tempo real desenvolvido com foco em acessibilidade e processamento de visão computacional temporal.

**Equipe:**
* 👤 Ana Clara Guimarães
* 👤 Mateus Bueno Ferreira
* 👤 Sofia Schmitz
* 👤 Vinícios Buzzi

---

---

## 📦 Estrutura Completa do Repositório

```plaintext
📁 S.I.N.A.I.S/
├── 📁 data/
│   ├── 📁 raw_videos/                  # Vídeos .mp4 baixados do INES (ignorado no git)
│   │   └── 📁 {ASSUNTO}/
│   │       └── 📁 {PALAVRA}/
│   │           └── 📄 video.mp4
│   └── 📁 processed_features/          # Matrizes .npy extraídas (ignorado no git)
│       └── 📁 {CLASSE}/
│           └── 📄 {video_id}.npy       # Shape: (30, 168)
│
├── 📁 models/
│   └── 📁 saved_weights/               # Pesos treinados (.pth) (ignorado no git)
│
├── 📁 scripts/
│   ├── 📄 download_ines_videos.py      # 🌐 Baixa vídeos do Dicionário INES
│   ├── 📄 inspect_ines_vocabulary.py   # 📊 Analisa vocabulário e gera CSV
│   ├── 📄 extract_landmarks.py         # (Fase 2) ETL: Vídeo → matrizes .npy
│   ├── 📄 data_augmentation.py         # (Fase 2) Ruído, espelho, zoom
│   └── 📄 train.py                     # (Fase 3) Treino/validação LSTM
│
├── 📄 app.py                           # 🎨 Dashboard Streamlit (MVP Interface)
├── 📄 requirements.txt                 # Dependências do projeto
├── 📄 .gitignore                       # Exclui vídeos e pesos grandes do git
└── 📄 README.md                        # Este arquivo
```

---

## 🌐 Fase 1: Coleta de Dados (Dicionário INES)

O dicionário do INES disponibiliza publicamente todos os vídeos de sinais via
o arquivo `palavras.js`, que embarca todo o vocabulário. Nossos scripts lêem
esse arquivo diretamente e baixam os vídeos organizados por assunto.

### Inspecionar o Vocabulário (sem baixar nada)

```bash
python scripts/inspect_ines_vocabulary.py
```

Gera estatísticas por assunto e exporta `data/vocabulario_ines.csv`.

### Baixar Todos os Vídeos

```bash
python scripts/download_ines_videos.py
```

> ⚠️ O dicionário tem mais de **7.000 palavras**. O download completo pode levar
> horas. Use `--assunto` para começar com um subconjunto menor.

### Baixar por Assunto Específico

```bash
# Apenas frutas (~20 palavras)
python scripts/download_ines_videos.py --assunto FRUTA

# Apenas animais
python scripts/download_ines_videos.py --assunto ANIMAL

# Sentimentos
python scripts/download_ines_videos.py --assunto SENTIMENTOS
```

**Assuntos disponíveis:**
`ALIMENTO/BEBIDA`, `ANIMAL/INSETO/PEIXE/AVE`, `CORPO`, `COR/FORMA`,
`ESPORTE/DIVERSÃO`, `FAMÍLIA`, `FRUTA`, `HIGIENE/SAÚDE`, `LEGUME/VERDURA`,
`PAÍS/ESTADO/CIDADE`, `PROFISSÃO/TRABALHO`, `SENTIMENTOS`, `TRANSPORTE/VEÍCULO`, `VESTUÁRIO/COMPLEMENTOS`

O script suporta **retomada automática**: se o download for interrompido,
retoma de onde parou consultando `data/download_log.csv`.

---

## 🚀 Como Executar o MVP (Fase 1 - Dashboard)

O MVP consiste em um dashboard interativo premium desenvolvido em Streamlit que captura a webcam do usuário em tempo real, executa o rastreamento tridimensional das mãos via **MediaPipe** e simula a interface de tradução com áudio sintetizado não-bloqueante (gTTS).

### 1. Pré-requisitos
*   Python 3.10 ou superior instalado.
*   Uma webcam conectada.

### 2. Configuração do Ambiente e Instalação
No terminal, na raiz do repositório, execute:

```bash
# 1. Criação do ambiente virtual
python -m venv venv

# 2. Ativação do ambiente
# No Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# No Windows (CMD):
.\venv\Scripts\activate.bat
# No Linux/macOS:
source venv/bin/activate

# 3. Instalação das dependências
pip install -r requirements.txt
```

### 3. Inicialização do Dashboard
Com o ambiente ativado, execute:

```bash
streamlit run app.py
```

O navegador abrirá automaticamente no endereço `http://localhost:8501`, exibindo a interface premium com o rastreador de landmarks funcionando perfeitamente de forma responsiva!

---

## 📚 Validação Linguística e Padrão Ouro
Todas as classes de gestos e transformações sintáticas devem seguir rigorosamente o **Dicionário do INES** (Instituto Nacional de Educação de Surdos).
