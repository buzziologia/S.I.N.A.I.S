# 📜 scripts/ — Pipeline de Processamento de Dados

Esta pasta contém todos os scripts utilitários do S.I.N.A.I.S, organizados por **fase do pipeline**. Execute-os **na ordem indicada abaixo**.

---

## 🗺️ Ordem de Execução

```
Fase 1 → Fase 2a → Fase 2b → Fase 3
```

| Fase | Script | O que faz |
|------|--------|-----------|
| **1a** | `inspect_ines_vocabulary.py` | Analisa e exporta o vocabulário do INES (sem baixar nada) |
| **1b** | `download_ines_videos.py` | Baixa vídeos e imagens de configuração de mão do INES |
| **2a** | `extract_landmarks.py` | Extrai landmarks MediaPipe dos vídeos → matrizes `.npy` |
| **2b** | `data_augmentation.py` | Gera variações aumentadas das matrizes `.npy` |
| **3**  | `train.py` | Treina e valida o classificador LSTM em PyTorch |

---

## 📋 Fase 1 — Coleta de Dados (INES)

### `inspect_ines_vocabulary.py`
Inspeciona o vocabulário disponível no Dicionário Digital do INES **sem baixar nada**.
Útil para decidir quais assuntos priorizar antes de iniciar o download.

```bash
# Mostra estatísticas por assunto e exporta data/vocabulario_ines.csv
python scripts/inspect_ines_vocabulary.py
```

**Saída:** `data/vocabulario_ines.csv`

---

### `download_ines_videos.py`
Baixa os 3 artefatos por palavra do Dicionário INES:
- `video.mp4` — execução do sinal em LIBRAS
- `configuracao_mao.jpg` — foto da configuração inicial da mão (classificação HSL)

> **Nota:** A imagem do sinal (sinal.jpg) não está disponível publicamente via URL direta no servidor do INES.

```bash
# Baixar tudo (pode demorar horas — 7.000+ palavras)
python scripts/download_ines_videos.py

# Baixar apenas um assunto específico (recomendado para começar)
python scripts/download_ines_videos.py --assunto SENTIMENTOS
python scripts/download_ines_videos.py --assunto FRUTA
python scripts/download_ines_videos.py --assunto ANIMAL
```

**Assuntos disponíveis:**
`ALIMENTO/BEBIDA`, `ANIMAL/INSETO/PEIXE/AVE`, `ANO SIDERAL`, `APARELHO/MÁQUINA`,
`CASA`, `COR/FORMA`, `CORPO`, `ESPORTE/DIVERSÃO`, `FAMÍLIA`, `FRUTA`,
`HIGIENE/SAÚDE`, `LEGUME/VERDURA`, `MATÉRIA/SUBSTÂNCIA`, `NUMERAL/DINHEIRO`,
`PAÍS/ESTADO/CIDADE`, `PLANTA/FLOR/NATUREZA`, `PROFISSÃO/TRABALHO`,
`SENTIMENTOS`, `TRANSPORTE/VEÍCULO`, `VESTUÁRIO/COMPLEMENTOS`

**Saída:**
```
data/raw_videos/{ASSUNTO}/{PALAVRA}/
    ├── {nome}.mp4
    └── configuracao_mao.jpg
```

**Retomada automática:** Se o download for interrompido, rode o mesmo comando novamente. O script verifica `data/download_log.csv` e pula o que já foi baixado.

---

## 🔬 Fase 2 — Extração de Características (ETL)

> ⚠️ **Pré-requisito:** Fase 1 concluída. Os vídeos devem estar em `data/raw_videos/`.

### `extract_landmarks.py` _(a implementar)_
Processa cada `.mp4` com **MediaPipe Holistic** e salva as coordenadas normalizadas.

```bash
python scripts/extract_landmarks.py
```

**O que faz internamente:**
1. Lê cada vídeo frame a frame com OpenCV
2. Extrai **168 coordenadas** por frame:
   - Mão esquerda: 21 landmarks × 3 (X, Y, Z) = **63 dims**
   - Mão direita: 21 landmarks × 3 (X, Y, Z) = **63 dims**
   - Pose (ombros + cotovelos): 4 landmarks × 3 = **12 dims**
   - Face (boca + sobrancelhas): 10 landmarks × 3 = **30 dims**
3. Normaliza cada landmark: translação ao pulso + escalonamento pela distância máxima
4. Se a mão não for detectada → preenche com `np.zeros(63)` (zero-padding)
5. Agrupa 30 frames consecutivos → tensor `(30, 168)`

**Saída:** `data/processed_features/{CLASSE}/{video_id}.npy` — shape `(30, 168)`

---

### `data_augmentation.py` _(a implementar)_
Aumenta artificialmente o dataset para melhorar a robustez do modelo.

```bash
python scripts/data_augmentation.py
```

**Transformações aplicadas no nível de coordenadas** (sem reprocessar vídeo):
| Transformação | Parâmetro |
|---|---|
| Ruído Gaussiano | σ = 0.005 |
| Zoom Virtual | fator entre 0.95 e 1.05 |
| Espelhamento Horizontal | troca mão esq ↔ mão dir |

> ⚠️ No espelhamento, os vetores de mão esquerda e direita devem ser **trocados entre si** para manter coerência anatômica.

---

## 🧠 Fase 3 — Treinamento do Modelo

> ⚠️ **Pré-requisito:** Fases 1 e 2 concluídas. As matrizes `.npy` devem estar em `data/processed_features/`.

### `train.py` _(a implementar)_
Treina o classificador LSTM em PyTorch com os dados extraídos.

```bash
python scripts/train.py
```

**Arquitetura:** `LibrasLSTMClassifier`
- Input: `(batch_size, 30, 168)`
- LSTM bidirecional: 2 camadas, `hidden_dim=128`, `dropout=0.3`
- Output: softmax sobre N classes

**Divisão de dados:** `GroupKFold` por ID de vídeo — **nunca** divisão aleatória de frames para evitar data leakage.

**Saída:** `models/saved_weights/libras_lstm.pth`
