"""
Experimento: a ideia de "N modelos por grupo + detectar OOD pra escolher o
modelo" funciona? Treina 2 modelos em grupos DISJUNTOS (FRUTA e COR_FORMA) e
compara 4 scores de OOD como ROTEADOR (escolher o modelo "mais in-distribution"):

  MSP     — max softmax (baseline)
  Energy  — -logsumexp(logits)
  KNN     — distância ao k-ésimo vizinho no banco de features de treino
  Maha    — distância de Mahalanobis ao centro de classe mais próximo

Todos convertidos para "ood_score" onde MENOR = mais in-distribution.

Métrica decisiva:
  - Oráculo  : acurácia se SOUBÉSSEMOS o grupo certo (sempre usa o modelo certo).
  - Combinador: acurácia roteando pelo modelo de menor ood_score.
Se o combinador << oráculo, a seleção é o gargalo — nenhum score salva.

Uso:
    python scripts/experimento_grupos.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # raiz → models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))        # scripts → train
from models.lstm_classifier import LIBRASClassifier
import train  # reusa LibrasDataset, splits e pré-processamento/aug

GRUPO_A = 'FRUTA'
GRUPO_B = 'COR_FORMA'
EPOCHS  = 50
K_KNN   = 5


class _Sub(Dataset):
    """Subset com o mesmo pré-processamento do train.py (aug só no treino)."""
    def __init__(self, base, indices, treino):
        self.base, self.indices, self.treino = base, indices, treino

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        caminho, label = self.base.amostras[self.indices[i]]
        t = np.load(caminho).astype(np.float32)
        t = train.paddar_ou_truncar(t)
        t = train.normalizar_landmarks(t)
        t = train.aumentar(t, treino=self.treino)
        return torch.from_numpy(t), label


def treinar_grupo(assunto, device):
    base = train.LibrasDataset('data/processed_features', 'data/raw_videos',
                               treino=False, assunto=assunto)
    tr, va, te = train._split_estratificado(base)
    loader_tr = DataLoader(_Sub(base, tr, True), batch_size=32, shuffle=True)

    modelo = LIBRASClassifier(input_dim=train.INPUT_DIM,
                              num_classes=len(base.classes)).to(device)
    opt  = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        modelo.train()
        for X, y in loader_tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            crit(modelo(X), y).backward()
            opt.step()
    print(f"  [{assunto}] {len(base.classes)} classes treinadas")
    return modelo, base, tr, te


def analisar(modelo, base, indices, device):
    """Roda o modelo (sem aug) e devolve conf, pred, label, energia e features."""
    confs, preds, labels, energias, feats = [], [], [], [], []
    modelo.eval()
    with torch.no_grad():
        for idx in indices:
            caminho, label = base.amostras[idx]
            t = train.normalizar_landmarks(train.paddar_ou_truncar(
                np.load(caminho).astype(np.float32)))
            x = torch.from_numpy(t).unsqueeze(0).to(device)
            out, _ = modelo.lstm(x)
            last   = out[:, -1, :]
            feat   = modelo.classifier[:5](last)          # penúltima camada (256)
            logits = modelo.classifier[5](feat)[0]
            probs  = torch.softmax(logits, dim=0)
            conf, pred = probs.max(0)
            confs.append(conf.item()); preds.append(pred.item()); labels.append(label)
            energias.append((-torch.logsumexp(logits, dim=0)).item())
            feats.append(feat[0].cpu().numpy())
    return {'conf': np.array(confs), 'pred': np.array(preds),
            'label': np.array(labels), 'energy': np.array(energias),
            'feat': np.array(feats, dtype=np.float64)}


# ── Scores de OOD baseados em features (menor = mais in-distribution) ──────────

def _l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

def _dist(A, B):
    a2 = (A ** 2).sum(1)[:, None]
    b2 = (B ** 2).sum(1)[None, :]
    return np.sqrt(np.maximum(a2 + b2 - 2 * A @ B.T, 0))

def knn_scores(feat, banco, k=K_KNN):
    D = _dist(_l2n(feat), _l2n(banco))
    D.sort(axis=1)
    return D[:, min(k, D.shape[1]) - 1]

def maha_fit(feat, label, shrink=0.5):
    classes = np.unique(label)
    means   = {c: feat[label == c].mean(0) for c in classes}
    centrado = feat - np.stack([means[l] for l in label])
    d = feat.shape[1]
    Sigma = centrado.T @ centrado / len(feat)
    # shrinkage forte: covariância de ~280 amostras em 256-d é mal-condicionada
    Sigma = (1 - shrink) * Sigma + shrink * (np.trace(Sigma) / d) * np.eye(d)
    return np.stack([means[c] for c in classes]), np.linalg.inv(Sigma)

def maha_scores(feat, M, Sinv):
    out = []
    for f in feat:
        diff = M - f
        out.append(np.einsum('cd,de,ce->c', diff, Sinv, diff).min())
    return np.array(out)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}")
    print(f"Treinando 2 modelos em grupos disjuntos ({EPOCHS} épocas cada)...")

    modelo_A, base_A, tr_A, te_A = treinar_grupo(GRUPO_A, device)
    modelo_B, base_B, tr_B, te_B = treinar_grupo(GRUPO_B, device)

    # Banco de features de treino + Gaussianas por modelo
    trA = analisar(modelo_A, base_A, tr_A, device)
    trB = analisar(modelo_B, base_B, tr_B, device)
    banco_A, maha_A = trA['feat'], maha_fit(trA['feat'], trA['label'])
    banco_B, maha_B = trB['feat'], maha_fit(trB['feat'], trB['label'])

    metodos = ['MSP', 'Energy', 'KNN', 'Maha']

    def processa(modelo_r, base_r, te_r, banco_r, mh_r, modelo_w, banco_w, mh_w):
        rgt = analisar(modelo_r, base_r, te_r, device)   # modelo certo
        wrg = analisar(modelo_w, base_r, te_r, device)   # modelo errado (OOD)
        correct = (rgt['pred'] == rgt['label']).astype(float)
        per = {
            'MSP':    (1 - rgt['conf'],                 1 - wrg['conf']),
            'Energy': (rgt['energy'],                   wrg['energy']),
            'KNN':    (knn_scores(rgt['feat'], banco_r), knn_scores(wrg['feat'], banco_w)),
            'Maha':   (maha_scores(rgt['feat'], *mh_r),  maha_scores(wrg['feat'], *mh_w)),
        }
        return per, correct

    perA, corrA = processa(modelo_A, base_A, te_A, banco_A, maha_A, modelo_B, banco_B, maha_B)
    perB, corrB = processa(modelo_B, base_B, te_B, banco_B, maha_B, modelo_A, banco_A, maha_A)
    correct = np.concatenate([corrA, corrB])
    oracle  = correct.mean()

    print(f"\nOráculo (sabe o grupo): {oracle:.1%}  | N={len(correct)} amostras de teste\n")
    print(f"{'Método':8s} | {'comb.acc':>8s} | {'errado venceu':>13s} | "
          f"{'P(certo|1)':>10s} | {'^155 (156 grupos)':>18s}")
    print('-' * 70)
    for m in metodos:
        r = np.concatenate([perA[m][0], perB[m][0]])   # ood do modelo certo
        w = np.concatenate([perA[m][1], perB[m][1]])   # ood do modelo errado
        certo_vence = r <= w
        comb        = float((certo_vence * correct).sum() / len(correct))
        p_certo     = float(certo_vence.mean())
        print(f"{m:8s} | {comb:7.1%} | {1 - p_certo:12.1%} | {p_certo:9.1%} | "
              f"{p_certo ** 155:17.2e}")

    print("\n── Veredito ──")
    print("  Compare 'comb.acc' com o oráculo: se todos ficam bem abaixo, a seleção")
    print("  é o gargalo. A coluna '^155' mostra a chance do modelo certo vencer os")
    print("  155 competidores no plano de 156 grupos — colapsa mesmo no melhor score.")


if __name__ == '__main__':
    main()
