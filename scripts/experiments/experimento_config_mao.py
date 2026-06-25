"""
Experimento A/B: a configuração de mão (configuracao_mao.jpg) como TAREFA AUXILIAR
no treino ajuda? (informação privilegiada — usada só no treino, ignorada na inferência)

- Baseline   : LSTM → cabeça de palavra (loss = CE da palavra).
- Multi-task : mesma rede + cabeça auxiliar que REGRIDE o vetor 126-d da configuração
               de mão canônica da palavra (loss = CE_palavra + λ·MSE_config).

Os dois compartilham init e ordem de batches (mesmo seed) p/ comparação justa.
Escopo: top-100 (rápido). Mede val_acc / test_acc.

────────────────────────────────────────────────────────────────────────────
RESULTADO (top-100, 40 épocas, λ=0.3):
    Baseline   : melhor val_acc=62.6% | test_acc=56.6%
    Multi-task : melhor val_acc=64.6% | test_acc=57.6%
    Δ test_acc : +1.0 pt  → ganho DESPREZÍVEL (dentro do ruído).

CONCLUSÃO: não compensa. A configuração de mão é constante por classe, então
só atua como regularizador fraco — não adiciona informação discriminativa nova.
Com 1 vídeo/palavra o gargalo é DADO (mais vídeos reais), não feature.
Mantido apenas como registro de experimento.
────────────────────────────────────────────────────────────────────────────

Uso:
    python scripts/experiments/experimento_config_mao.py --epochs 40 --lmbda 0.3
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))        # raiz → models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))              # scripts → train
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))  # scripts/training → extract_features
from models.lstm_classifier import LIBRASClassifier
import train
import extract_features as ef
import mediapipe as mp

SEED = 0


class _Sub(Dataset):
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


def carregar_handshapes(classes, dir_videos='data/raw_videos'):
    """Para cada classe: vetor 126-d da configuracao_mao.jpg (normalizado). Mascara o que faltar."""
    cset = {c: i for i, c in enumerate(classes)}
    alvo = np.zeros((len(classes), 126), dtype=np.float32)
    mask = np.zeros(len(classes), dtype=bool)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                     min_detection_confidence=0.3)
    for root, _, files in os.walk(dir_videos):
        palavra = os.path.basename(root)
        if palavra in cset and 'configuracao_mao.jpg' in files:
            img = cv2.imread(os.path.join(root, 'configuracao_mao.jpg'))
            if img is None:
                continue
            img = cv2.flip(img, 1)   # mesma convenção do extract_features
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                v = train.normalizar_landmarks(ef.frame_para_vetor(res).reshape(1, 126))[0]
                alvo[cset[palavra]] = v
                mask[cset[palavra]] = True
    hands.close()
    print(f"[CONFIG] handshape em {mask.sum()}/{len(classes)} classes")
    return torch.from_numpy(alvo), torch.from_numpy(mask)


class MultiTask(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        torch.manual_seed(SEED)                       # init idêntica ao baseline
        self.base = LIBRASClassifier(input_dim=126, num_classes=num_classes)
        self.aux  = nn.Linear(self.base.lstm.hidden_size * 2, 126)  # cabeça auxiliar

    def forward(self, x):
        out, _ = self.base.lstm(x)
        last   = out[:, -1, :]
        return self.base.classifier(last), self.aux(last)


def avaliar(modelo, loader, device, multitask):
    modelo.eval()
    acertos = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = modelo(X)[0] if multitask else modelo(X)
            acertos += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return acertos / total if total else 0.0


def treinar(multitask, dataset, tr, va, te, num_classes, args, device,
            alvo=None, mask=None):
    torch.manual_seed(SEED); np.random.seed(SEED)     # mesma ordem de batches/aug
    loader_tr = DataLoader(_Sub(dataset, tr, True),  batch_size=32, shuffle=True)
    loader_va = DataLoader(_Sub(dataset, va, False), batch_size=64)
    loader_te = DataLoader(_Sub(dataset, te, False), batch_size=64)

    modelo = MultiTask(num_classes).to(device) if multitask else \
        (torch.manual_seed(SEED), LIBRASClassifier(input_dim=126, num_classes=num_classes).to(device))[1]
    opt  = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    ce   = nn.CrossEntropyLoss()
    mse  = nn.MSELoss()

    melhor_val, test_no_melhor = 0.0, 0.0
    for ep in range(1, args.epochs + 1):
        modelo.train()
        for X, y in loader_tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            if multitask:
                logits, cfg = modelo(X)
                loss = ce(logits, y)
                m = mask[y]
                if m.any():
                    loss = loss + args.lmbda * mse(cfg[m], alvo[y][m])
            else:
                loss = ce(modelo(X), y)
            loss.backward()
            opt.step()
        val = avaliar(modelo, loader_va, device, multitask)
        if val >= melhor_val:
            melhor_val = val
            test_no_melhor = avaliar(modelo, loader_te, device, multitask)
        if ep % 5 == 0 or ep == args.epochs:
            print(f"  [{'MT ' if multitask else 'base'}] ep {ep:02d}/{args.epochs}  val_acc={val:.3f}")
    return melhor_val, test_no_melhor


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device} | top_k={args.top_k} | epochs={args.epochs} | λ={args.lmbda}")

    dataset = train.LibrasDataset('data/processed_features', 'data/raw_videos',
                                  treino=False, top_k=args.top_k)
    num_classes = len(dataset.classes)
    tr, va, te = train._split_estratificado(dataset)
    print(f"Classes: {num_classes} | treino {len(tr)} | val {len(va)} | teste {len(te)}")

    alvo, mask = carregar_handshapes(dataset.classes)
    alvo, mask = alvo.to(device), mask.to(device)

    print("\n── BASELINE (só palavra) ──")
    b_val, b_test = treinar(False, dataset, tr, va, te, num_classes, args, device)
    print("\n── MULTI-TASK (palavra + configuração de mão) ──")
    m_val, m_test = treinar(True, dataset, tr, va, te, num_classes, args, device, alvo, mask)

    print("\n" + "=" * 50)
    print(f"  BASELINE   : melhor val_acc={b_val:.1%} | test_acc={b_test:.1%}")
    print(f"  MULTI-TASK : melhor val_acc={m_val:.1%} | test_acc={m_test:.1%}")
    print(f"  Δ test_acc : {(m_test - b_test) * 100:+.1f} pontos")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lmbda', type=float, default=0.3, help='peso da loss auxiliar')
    main(parser.parse_args())
