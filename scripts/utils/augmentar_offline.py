"""
Augmentação offline - gera N variações de cada .npy em data/augmented_features/.

Para cada arquivo em data/processed_features/ são criadas N cópias augmentadas,
cada uma com uma semente aleatória diferente. O dataset de treino passa a ter
(1 + N) amostras por classe, permitindo validação real.

Uso:
    python scripts/augmentar_offline.py             # padrão: 9 cópias → 10 amostras/classe
    python scripts/augmentar_offline.py --n 19      # 19 cópias → 20 amostras/classe
    python scripts/augmentar_offline.py --n 4       # 4 cópias → 5 amostras/classe (rápido)
"""
import argparse
import os
import random
import numpy as np
from tqdm import tqdm

DIR_ORIG = 'data/processed_features'
DIR_AUG  = 'data/augmented_features'

# ── Augmentações (mesmo conjunto do train.py) ─────────────────────────────────

def ruido(t, sigma=0.02):
    return t + np.random.randn(*t.shape).astype(np.float32) * sigma

def escala(t, lo=0.8, hi=1.2):
    return t * np.random.uniform(lo, hi)

def deslocamento_xy(t, delta=0.05):
    d = t.copy()
    dx, dy = np.random.uniform(-delta, delta), np.random.uniform(-delta, delta)
    d[:, 0::3] += dx
    d[:, 1::3] += dy
    return d

def time_warp(t, max_fator=0.3):
    n = len(t)
    if n < 4:
        return t
    fator = 1.0 + np.random.uniform(-max_fator, max_fator)
    n_novo = max(4, int(n * fator))
    idxs   = np.linspace(0, n - 1, n_novo)
    orig   = np.arange(n)
    warped = np.array([np.interp(idxs, orig, t[:, col]) for col in range(t.shape[1])]).T
    if n_novo >= n:
        return warped[-n:]
    pad = np.zeros((n - n_novo, t.shape[1]), dtype=np.float32)
    return np.vstack([pad, warped])

def flip_horizontal(t):
    # Espelha horizontalmente: inverte X de cada mão presente e troca os slots
    # Left/Right (um espelho real troca a mão esquerda pela direita).
    # Features cru do MediaPipe (X em 0..1) → espelho é 1 - X.
    d = t.copy()
    for h in range(2):
        sl = slice(h * 63, (h + 1) * 63)
        bloco = d[:, sl]
        presente = ~np.all(bloco == 0, axis=1)   # não mexe em mão ausente (zeros)
        bloco[presente, 0::3] = 1.0 - bloco[presente, 0::3]
    # Troca os dois blocos de mão
    esquerda = d[:, 0:63].copy()
    d[:, 0:63]   = d[:, 63:126]
    d[:, 63:126] = esquerda
    return d

def shift_temporal(t, max_shift=4):
    n = len(t)
    s = np.random.randint(-max_shift, max_shift + 1)
    if s == 0:
        return t
    if s > 0:                         # adianta: descarta início
        return np.vstack([t[s:], np.zeros((s, t.shape[1]), dtype=np.float32)])
    else:                             # atrasa: descarta fim
        s = -s
        return np.vstack([np.zeros((s, t.shape[1]), dtype=np.float32), t[:-s]])

def aumentar(t):
    """Aplica um subconjunto aleatório de augmentações ao tensor (N, 126)."""
    t = t.astype(np.float32)
    t = ruido(t)
    if random.random() < 0.8:
        t = escala(t)
    if random.random() < 0.7:
        t = deslocamento_xy(t)
    if random.random() < 0.6:
        t = time_warp(t)
    if random.random() < 0.5:
        t = flip_horizontal(t)
    if random.random() < 0.5:
        t = shift_temporal(t)
    return t

# ── Main ──────────────────────────────────────────────────────────────────────
def main(n_copies: int):
    os.makedirs(DIR_AUG, exist_ok=True)

    arquivos = [f for f in os.listdir(DIR_ORIG) if f.endswith('.npy')]
    print(f'Arquivos originais: {len(arquivos)}')
    print(f'Cópias por arquivo: {n_copies}')
    print(f'Total gerado:       {len(arquivos) * n_copies}')
    print(f'Total com original: {len(arquivos) * (n_copies + 1)}')
    print(f'Destino:            {DIR_AUG}')
    print()

    ja_existem = len([f for f in os.listdir(DIR_AUG) if f.endswith('.npy')])
    if ja_existem > 0:
        print(f'  [!] {ja_existem} arquivos já existem em {DIR_AUG}.')
        resp = input('  Regenerar tudo? (s/N): ').strip().lower()
        if resp != 's':
            print('  Cancelado.')
            return

    gerados = 0
    for arq in tqdm(arquivos, desc='Augmentando', unit='sinal'):
        stem = os.path.splitext(arq)[0]
        orig = np.load(os.path.join(DIR_ORIG, arq))

        for i in range(n_copies):
            aug  = aumentar(orig)
            nome = f'{stem}_aug{i:02d}.npy'
            np.save(os.path.join(DIR_AUG, nome), aug)
            gerados += 1

    print(f'\n✓ {gerados} arquivos gerados em {DIR_AUG}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augmentação offline de features Libras')
    parser.add_argument('--n', type=int, default=9,
                        help='Número de cópias augmentadas por sinal (padrão: 9)')
    args = parser.parse_args()
    main(args.n)
