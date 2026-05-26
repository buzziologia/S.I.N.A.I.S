# 🧠 models/ — Modelos de Classificação

Esta pasta armazena a **definição da arquitetura** e os **pesos treinados** do classificador de sinais LIBRAS.

---

## 📁 Estrutura Esperada

```
models/
├── lstm_classifier.py    ← (a definir) Arquitetura do modelo
└── saved_weights/
    └── *.pth / *.h5      ← (gerado pelo train.py) Pesos treinados
```

> `saved_weights/` está no `.gitignore` — arquivos de pesos binários não vão ao repositório.

---

## 📌 A Definir pelo Grupo

- Arquitetura do modelo (LSTM, Transformer, etc.)
- Hiperparâmetros de treinamento
- Estratégia de divisão de dados (train/val/test)
- Formato de serialização dos pesos

Consulte `scripts/train.py` e `scripts/extract_landmarks.py` para entender o formato de entrada esperado pelo modelo: tensores de shape `(batch_size, 30, 168)`.
