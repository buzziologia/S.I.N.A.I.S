# 🤟 S.I.N.A.I.S - Sistema Integrado de Tradução e Processamento de Sinais

**S.I.N.A.I.S** é um tradutor de LIBRAS (Língua Brasileira de Sinais) vídeo → texto em tempo real: a webcam captura o sinal, o MediaPipe extrai os landmarks das mãos e uma LSTM bidirecional classifica a palavra.

**Equipe:**
* 👤 Ana Clara Guimarães
* 👤 Mateus Bueno Ferreira
* 👤 Sofia Schmitz
* 👤 Vinícios Buzzi

---

## Como funciona

```
vídeo → MediaPipe Hands (126 features/frame = 2 mãos × 21 landmarks × XYZ)
      → recorte da janela de atividade + normalização (models/preprocess.py)
      → LSTM bidirecional (models/lstm_classifier.py)
      → palavra
```

Cada frame vira um vetor de 126 valores: a mão rotulada `Left` pelo MediaPipe ocupa as posições 0:63 e a `Right` as posições 63:126 (mão ausente = zeros). As sequências são recortadas para a janela onde há mão detectada, ajustadas para 30 frames e normalizadas (translação ao pulso + escala) - **todo o pré-processamento vive em um único módulo (`models/preprocess.py`)**, compartilhado por treino, avaliação e câmera.

---

## 📦 Estrutura do Repositório

```plaintext
📁 S.I.N.A.I.S/
├── 📁 data/                          # Dados locais (ignorados no git)
│   ├── 📁 raw_videos/                # Vídeos .mp4 do INES (ASSUNTO/PALAVRA/)
│   ├── 📁 processed_features/        # Matrizes .npy extraídas dos vídeos INES
│   ├── 📁 augmented_features/        # Variações offline (augmentar_offline.py)
│   ├── 📁 meus_videos/ + meus_features/           # Vídeos próprios (treino)
│   ├── 📁 meus_videos_holdout/ + meus_features_holdout/  # Vídeos próprios (teste)
│   └── 📄 palavras_por_frequencia.csv # Ranking de uso das palavras (PT-BR)
├── 📁 models/
│   ├── 📄 preprocess.py              # Pré-processamento ÚNICO (treino = inferência)
│   ├── 📄 lstm_classifier.py         # LSTM bidirecional
│   └── 📁 saved_weights/             # Checkpoints: {id}_modelo.pth + {id}_classes.json
├── 📁 scripts/
│   ├── 📄 train.py                   # Treino
│   ├── 📄 avaliar_holdout.py         # Avaliação no hold-out (métrica honesta)
│   ├── 📄 testar_camera.py           # Reconhecimento ao vivo pela webcam
│   ├── 📁 training/
│   │   ├── 📄 extract_features.py    # Vídeo .mp4 → matriz .npy (N, 126)
│   │   └── 📄 coletar_videos.py      # Coleta de vídeos próprios pela webcam
│   ├── 📁 utils/
│   │   ├── 📄 download_ines_videos.py # Baixa o Dicionário INES
│   │   ├── 📄 augmentar_offline.py    # N variações por .npy
│   │   └── 📄 ranquear_palavras.py    # Ranking de frequência PT-BR
│   └── 📁 experiments/README.md      # Registro de experimentos (resultados)
├── 📄 app.py                         # Dashboard Streamlit (rastreamento de mãos)
├── 📄 justfile                       # Atalhos de comandos (just <comando>)
└── 📄 requirements.txt
```

---

## ⚙️ Setup do Ambiente

O projeto exige **Python 3.11** (o `mediapipe ≥ 0.10.30` removeu a API `solutions.hands` usada aqui). Recomendado via conda:

```bash
conda create -n sinais311 python=3.11
conda activate sinais311

# PyTorch CPU (opcional, reduz o download de 2GB+ para ~150MB)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

No Windows, defina `PYTHONUTF8=1` (variável de ambiente do usuário) para a saída dos scripts não quebrar acentos.

Opcional: instale o [just](https://github.com/casey/just) (`winget install Casey.Just`) para usar os atalhos do `justfile` - os comandos abaixo mostram as duas formas. Aponte o `just` para o seu Python definindo `SINAIS_PYTHON` (ex.: `$env:SINAIS_PYTHON = "python"` no PowerShell, com o env conda ativado).

---

## 🚀 Pipeline completo

### 1. Baixar os vídeos do Dicionário INES

```bash
just download          # ou: python scripts/utils/download_ines_videos.py
# só um assunto:       python scripts/utils/download_ines_videos.py --assunto FRUTA
```

Baixa vídeo + imagem do sinal + configuração de mão por palavra para `data/raw_videos/{ASSUNTO}/{PALAVRA}/`. Suporta **retomada automática** (log em `data/download_log.csv`).

### 2. Extrair features dos vídeos

```bash
just extract           # ou: python scripts/training/extract_features.py
```

Converte cada `.mp4` em uma matriz `(N_frames, 126)` salva em `data/processed_features/`. Pula arquivos já processados.

### 3. Ranquear o vocabulário por frequência de uso

```bash
just ranquear          # gera data/palavras_por_frequencia.csv
```

Permite treinar só nas K palavras mais usadas do PT-BR (`--top_k`).

### 4. Augmentação offline

```bash
just augment 9         # 9 variações por .npy → 10 amostras por palavra
```

Necessária quando há 1 vídeo por palavra: sem ela não existe divisão treino/validação/teste.

### 5. Coletar vídeos próprios (essencial para generalizar)

```bash
just coletar --top_k 10 --reps 6 --holdout 2
```

Grava você sinalizando pela webcam, com vídeo de referência do INES na tela, dicas de variação por repetição e revisão de cada take. As últimas `--holdout` reps vão para `data/meus_features_holdout/` (nunca entram no treino → avaliação honesta).

> **Por quê:** o modelo treinado só com os vídeos do INES (1 sinalizante) não generaliza para outras pessoas (*domain shift*). No experimento com as 10 palavras mais frequentes, incluir 4 repetições próprias por palavra elevou a acurácia no hold-out de **35% para 90%**.

### 6. Treinar

```bash
just train --top_k 10                                   # só vídeos INES
just train --top_k 10 --dir_meus data/meus_features     # INES + vídeos próprios
```

Salva `models/saved_weights/{timestamp}_modelo.pth` + `{timestamp}_classes.json`. Early stopping pela loss de validação.

### 7. Avaliar no hold-out

```bash
just avaliar --pesos models/saved_weights/XXX_modelo.pth        # hold-out próprio
just avaliar --pesos models/saved_weights/XXX_modelo.pth --raw  # vídeos INES
```

Relatório por take (top-3 predições), por palavra e acurácia total.

### 8. Testar ao vivo pela webcam

```bash
just demo      # melhor modelo atual (top-10, 3 sinalizantes - 98% no hold-out)
# ou qualquer outro checkpoint:
just camera --pesos models/saved_weights/XXX_modelo.pth
```

Mostra o sinal reconhecido, a confiança, o vídeo de referência da palavra e rejeita entradas fora do vocabulário (detector OOD por KNN, cacheado em disco). `--sem_ood` desativa a rejeição; `--confianca` ajusta o limiar. Variantes do dicionário (QUE1/QUE2) são exibidas unificadas como "QUE".

---

## 📚 Padrão de Referência

Todos os sinais seguem o **Dicionário Oficial de LIBRAS do INES** (Instituto Nacional de Educação de Surdos): [dicionario.ines.gov.br](https://dicionario.ines.gov.br/)
