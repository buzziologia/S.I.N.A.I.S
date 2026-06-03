# 🤟 S.I.N.A.I.S - Sistema Integrado de Tradução e Processamento de Sinais

**S.I.N.A.I.S** é um tradutor de LIBRAS em tempo real desenvolvido com foco em acessibilidade e visão computacional.

**Equipe:**
* 👤 Ana Clara Guimarães
* 👤 Mateus Bueno Ferreira
* 👤 Sofia Schmitz
* 👤 Vinícios Buzzi

---

## 📦 Estrutura do Repositório

```plaintext
📁 S.I.N.A.I.S/
├── 📁 data/                       # Dados locais (ver data/README.md)
│   ├── 📁 raw_videos/             # Vídeos .mp4 do INES (ignorado no git)
│   └── 📁 processed_features/     # Matrizes .npy extraídas (ignorado no git)
├── 📁 models/                     # Modelo e pesos treinados (ver models/README.md)
│   └── 📁 saved_weights/          # Pesos treinados (ignorado no git)
├── 📁 scripts/                    # Scripts de ETL e treino (ver scripts/README.md)
│   ├── 📄 download_ines_videos.py # ✅ Baixa vídeos do Dicionário INES
│   └── 📄 inspect_ines_vocabulary.py # ✅ Analisa vocabulário
│   └── 📄 extract_features.py        # ✅ Extrai matrizes temporais (.npy) dos vídeos
├── 📄 app.py                      # ✅ Dashboard Streamlit (MVP)
├── 📄 requirements.txt            # Dependências do projeto
└── 📄 .gitignore
```

---

## ⚙️ Setup do Ambiente

```bash
# 1. Criar ambiente virtual
python -m venv venv

# 2. Ativar
.\\venv\\Scripts\\Activate.ps1   # Windows PowerShell
source venv/bin/activate         # Linux/macOS

# 3. Atualizar pip, setuptools e wheel (evita erros de compilação)
pip install --upgrade pip setuptools wheel

# 4. Instalar PyTorch versão CPU (opcional - MUITO recomendado: reduz o download de 2GB+ para ~150MB)
# Se quiser a versão padrão com GPU/CUDA completa, pule esta linha.
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Instalar as demais dependências do projeto
pip install -r requirements.txt
```

---

## ✅ O que está implementado

### Fase 1 — Coleta de Dados (Dicionário INES)

Baixa vídeos de sinais LIBRAS e imagens de configuração de mão diretamente do Dicionário Digital do INES.

```bash
# Inspecionar vocabulário disponível (sem baixar nada)
python scripts/inspect_ines_vocabulary.py

# Baixar um assunto específico
python scripts/download_ines_videos.py --assunto SENTIMENTOS

# Baixar tudo (7.000+ palavras — pode demorar horas)
python scripts/download_ines_videos.py
```

> O download suporta **retomada automática**: se interrompido, rode o mesmo comando novamente.

Consulte [`scripts/README.md`](scripts/README.md) para a lista completa de assuntos e detalhes.

---

### Fase 2 — Processamento de Dados (Visão Computacional)
Extrai os pontos-chave (landmarks) espaciais e temporais das mãos utilizando o MediaPipe. Transforma a dinâmica dos vídeos .mp4 em vetores matemáticos para o treinamento da IA.

```bash
# Extrair landmarks de todos os vídeos baixados
python scripts/extract_features.py
```
Lê os arquivos de vídeo em data/raw_videos/ e gera matrizes isoladas em data/processed_features/{nome}.npy.

---

### MVP — Dashboard de Rastreamento (app.py)

Interface Streamlit que captura a webcam em tempo real e rastreia os landmarks das mãos via MediaPipe.

```bash
streamlit run app.py
```

Acesse em `http://localhost:8501`.

---

## 🔜 Próximas Fases (A Definir)

- **Fase 3:** Treinamento do classificador temporal
- **Fase 4:** Integração do modelo ao dashboard com tradução e síntese de voz

---

## 📚 Padrão de Referência

Todos os sinais seguem o **Dicionário Oficial de LIBRAS do INES** (Instituto Nacional de Educação de Surdos): [dicionario.ines.gov.br](https://dicionario.ines.gov.br/)
