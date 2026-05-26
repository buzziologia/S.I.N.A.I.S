# 📂 data/ — Armazenamento de Dados do Pipeline

Esta pasta contém **todos os dados do projeto**, organizados em duas subpastas com responsabilidades distintas. Ambas estão no `.gitignore` — os dados **nunca vão para o repositório** (são pesados demais e regeneráveis).

---

## ⚠️ Aviso Importante

```
data/raw_videos/       → ignorado pelo git (vídeos binários pesados)
data/processed_features/ → ignorado pelo git (matrizes numpy geradas)
data/download_log.csv  → ignorado pelo git (log local de progresso)
```

Para obter os dados, execute os scripts da pasta `scripts/` conforme documentado em [`scripts/README.md`](../scripts/README.md).

---

## 📁 `raw_videos/` — Vídeos Brutos do INES

Contém os vídeos `.mp4` e imagens de configuração de mão baixados do [Dicionário Digital do INES](https://dicionario.ines.gov.br/).

**Estrutura:**
```
raw_videos/
└── {ASSUNTO}/
    └── {PALAVRA}/
        ├── {arquivo}.mp4          ← vídeo do sinal em LIBRAS
        └── configuracao_mao.jpg   ← foto da configuração da mão (hand shape)
```

**Exemplo real:**
```
raw_videos/
└── SENTIMENTOS/
    ├── AMOR/
    │   ├── amor1Sm_Prog001.mp4
    │   └── configuracao_mao.jpg
    └── ALEGRIA1/
        ├── alegria1Sm_Prog001.mp4
        └── configuracao_mao.jpg
```

**Como popular esta pasta:**
```bash
# Baixar um assunto específico
python scripts/download_ines_videos.py --assunto SENTIMENTOS

# Verificar progresso
cat data/download_log.csv
```

---

## 📁 `processed_features/` — Matrizes de Features Extraídas

Contém as matrizes NumPy geradas pelo script `extract_landmarks.py` (Fase 2).
Cada arquivo representa **uma sequência de 30 frames** de um sinal, já normalizada e pronta para o treinamento.

**Estrutura:**
```
processed_features/
└── {CLASSE}/
    └── {video_id}.npy    ← tensor de shape (30, 168)
```

**Exemplo:**
```
processed_features/
└── AMOR/
    ├── amor1Sm_Prog001.npy        ← shape: (30, 168)
    ├── amor1Sm_Prog001_aug1.npy   ← shape: (30, 168) [augmentado]
    └── amor1Sm_Prog001_aug2.npy   ← shape: (30, 168) [augmentado]
```

**Dimensões do tensor `(30, 168)`:**

| Índice | Região | Dimensões |
|--------|--------|-----------|
| `[0:63]` | Mão Esquerda (21 landmarks × XYZ) | 63 |
| `[63:126]` | Mão Direita (21 landmarks × XYZ) | 63 |
| `[126:138]` | Pose — Ombros (11, 12) + Cotovelos (13, 14) | 12 |
| `[138:168]` | Face — Boca + Sobrancelhas (10 landmarks) | 30 |
| **Total** | | **168** |

> **Normalização aplicada:** Translação ao pulso (landmark 0) + escalonamento pela distância máxima ao dedo médio (landmark 12). Mão ausente → `np.zeros(63)`.

**Como popular esta pasta:**
```bash
python scripts/extract_landmarks.py     # extrai landmarks
python scripts/data_augmentation.py     # gera variações
```

---

## 📄 `download_log.csv` — Log de Progresso do Download

Arquivo gerado automaticamente pelo `download_ines_videos.py`. Registra o status de cada palavra baixada para permitir **retomada automática** do download.

**Colunas:**

| Coluna | Descrição |
|--------|-----------|
| `ident` | ID único da palavra no INES |
| `palavra` | Nome do sinal (ex: `AMOR`) |
| `video_file` | Nome do arquivo `.mp4` |
| `image_file` | Nome do arquivo de imagem do sinal (pode ser N/A no servidor) |
| `hand_file` | Nome do arquivo `cg*.jpg` da configuração de mão |
| `status` | `OK` ou `ERRO` |
| `message` | Detalhes (ex: `imagem=OK \| mao=OK`) |
