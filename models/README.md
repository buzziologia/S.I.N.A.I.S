# 🧠 models/ — Definição e Pesos do Classificador

Esta pasta contém a **arquitetura do modelo de Deep Learning** e os **pesos treinados** do classificador temporal de sinais LIBRAS.

---

## 📁 Estrutura

```
models/
├── lstm_classifier.py    ← (a criar) Arquitetura do modelo em PyTorch
└── saved_weights/
    └── libras_lstm.pth   ← (gerado pelo train.py) Pesos treinados
```

> `saved_weights/*.pth` está no `.gitignore` — pesos binários não vão ao repositório.

---

## 🏗️ Arquitetura: `LibrasLSTMClassifier`

O classificador utiliza uma **rede LSTM** (Long Short-Term Memory) para capturar a dependência temporal nos sinais de LIBRAS.

**Arquivo a criar:** `models/lstm_classifier.py`

```python
import torch
import torch.nn as nn

class LibrasLSTMClassifier(nn.Module):
    """
    Classificador temporal para sequências de landmarks LIBRAS.

    Input:  (batch_size, sequence_length=30, n_features=168)
    Output: (batch_size, n_classes)
    """
    def __init__(self, input_dim: int = 168, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=False  # False para baixa latência em tempo real
        )
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc_block(out[:, -1, :])  # último timestep
```

---

## ⚙️ Hiperparâmetros Recomendados

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `input_dim` | `168` | Vetor de features por frame (schema fixo) |
| `hidden_dim` | `128` | Capacidade suficiente sem overfitting |
| `num_layers` | `2` | Hierarquia temporal em dois níveis |
| `dropout` | `0.3` | Regularização entre camadas LSTM |
| `sequence_length` | `30` | ~1 segundo a 30 FPS |
| `batch_size` | `32` | Balanço entre memória e convergência |

---

## 🎯 Regras Críticas de Treinamento

### Divisão de Dados (GroupKFold)
**NUNCA** divida aleatoriamente por frames. Sempre divida **por ID de vídeo**:

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
# groups = array de video_ids (garante que frames do mesmo vídeo
# ficam todos no mesmo split — evita data leakage)
for train_idx, val_idx in gkf.split(X, y, groups=video_ids):
    ...
```

### Inferência em Tempo Real (Debounce)
Para evitar "piscadas" na tradução da interface:

```python
CONFIDENCE_THRESHOLD = 0.80  # confiança mínima
MIN_CONSECUTIVE_FRAMES = 5   # frames consecutivos com mesma predição

# Só exibe e aciona áudio após 5 frames seguidos com a mesma classe
# e confiança > 80%
```

---

## 💾 Carregar e Salvar Pesos

```bash
# Treinar e salvar (via script)
python scripts/train.py
# → salva em models/saved_weights/libras_lstm.pth

# Carregar para inferência (no app.py)
model = LibrasLSTMClassifier(num_classes=N)
model.load_state_dict(torch.load("models/saved_weights/libras_lstm.pth"))
model.eval()
```
