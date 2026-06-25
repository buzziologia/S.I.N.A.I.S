"""
Script de diagnóstico do pipeline de treino.

Roda 4 testes em ordem crescente de complexidade:

  Teste 1 — Sanidade do modelo
    O modelo consegue overfitar 1 único batch?
    Se não → problema na arquitetura ou no loop de treino.

  Teste 2 — Sanidade dos dados
    Os .npy têm conteúdo variado? As classes estão mapeadas corretamente?
    Se não → problema no extract_features.py ou no Dataset.

  Teste 3 — Overfit intencional (10 classes, sem val)
    O modelo consegue memorizar 10 amostras reais?
    Se não → modelo muito fraco ou features sem informação.

  Teste 4 — Mini-treino real (50 classes, com val)
    O modelo generaliza minimamente para dados não vistos?
    Se não → problema de dados (1 amostra/classe, overfitting estrutural).

Uso:
    python scripts/debug_treino.py
"""
import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from models.lstm_classifier import LIBRASClassifier

DIR_FEATURES = 'data/processed_features'
DIR_AUG      = 'data/augmented_features'
DIR_VIDEOS   = 'data/raw_videos'
MAX_SEQ_LEN  = 30
DIM_MAO      = 63                  # 1 mão (21 landmarks × XYZ)
NUM_MAOS     = 2                   # sinais bimanuais → 2 mãos
INPUT_DIM    = DIM_MAO * NUM_MAOS  # 126
SEP = '─' * 55

# ── helpers ──────────────────────────────────────────────────────────────────

def log(msg, ok=None):
    # ok=None → neutro; senão usa valor-verdade (robusto a numpy.bool_)
    icone = '·' if ok is None else ('✓' if bool(ok) else '✗')
    print(f'  [{icone}] {msg}')

def paddar(tensor):
    n = len(tensor)
    if n >= MAX_SEQ_LEN:
        return tensor[-MAX_SEQ_LEN:].astype(np.float32)
    pad = np.zeros((MAX_SEQ_LEN - n, INPUT_DIM), dtype=np.float32)
    return np.vstack([pad, tensor]).astype(np.float32)

def normalizar(tensor):
    """Normaliza cada mão de cada frame de forma independente (relativa ao pulso)."""
    out = tensor.copy()
    for i, frame in enumerate(out):
        for h in range(NUM_MAOS):
            ini, fim = h * DIM_MAO, (h + 1) * DIM_MAO
            pts = frame[ini:fim].reshape(21, 3)
            if np.all(pts == 0):
                continue
            pts = pts - pts[0]
            escala = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
            out[i, ini:fim] = (pts / escala).flatten()
    return out

def construir_mapa(dir_videos):
    mapa = {}
    for assunto in os.listdir(dir_videos):
        pasta_a = os.path.join(dir_videos, assunto)
        if not os.path.isdir(pasta_a): continue
        for palavra in os.listdir(pasta_a):
            pasta_p = os.path.join(pasta_a, palavra)
            if not os.path.isdir(pasta_p): continue
            for arq in os.listdir(pasta_p):
                if arq.endswith('.mp4'):
                    mapa[os.path.splitext(arq)[0]] = palavra
    return mapa

def carregar_amostras(max_classes=None, max_por_classe=None, incluir_aug=False):
    """
    Carrega (caminho, classe) de processed_features e, se incluir_aug=True e a
    pasta existir, também de augmented_features. Para arquivos augmentados
    (stem 'nomevideo_aug03') a classe é buscada pelo prefixo antes de '_aug',
    igual ao train.py.
    """
    mapa = construir_mapa(DIR_VIDEOS)
    amostras = []
    pastas = [DIR_FEATURES]
    if incluir_aug and os.path.isdir(DIR_AUG):
        pastas.append(DIR_AUG)
    for pasta in pastas:
        for arq in os.listdir(pasta):
            if not arq.endswith('.npy'): continue
            stem       = os.path.splitext(arq)[0]
            stem_busca = stem.split('_aug')[0] if '_aug' in stem else stem
            classe     = mapa.get(stem_busca)
            if classe is None: continue
            amostras.append((os.path.join(pasta, arq), classe))

    # Filtra classes com amostras suficientes
    contagem = Counter(c for _, c in amostras)
    classes_validas = sorted(set(c for c in contagem if contagem[c] >= 1))
    if max_classes:
        classes_validas = classes_validas[:max_classes]
    class_to_idx = {c: i for i, c in enumerate(classes_validas)}

    filtradas = [(p, class_to_idx[c]) for p, c in amostras if c in class_to_idx]
    if max_por_classe:
        por_classe = {}
        for p, l in filtradas:
            por_classe.setdefault(l, []).append((p, l))
        filtradas = [item for lst in por_classe.values() for item in lst[:max_por_classe]]

    return filtradas, classes_validas

def tensores_para_dataset(amostras):
    Xs, ys = [], []
    for caminho, label in amostras:
        t = paddar(normalizar(np.load(caminho)))
        Xs.append(t)
        ys.append(label)
    X = torch.tensor(np.array(Xs), dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    return TensorDataset(X, y)

# ── TESTE 1 — Sanidade do modelo ─────────────────────────────────────────────

def teste_1_sanidade_modelo():
    print(f'\n{SEP}')
    print('TESTE 1 — Sanidade do modelo (overfit de 1 batch sintético)')
    print(SEP)

    num_classes = 10
    modelo   = LIBRASClassifier(input_dim=INPUT_DIM, num_classes=num_classes)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    # Batch sintético fixo (sempre os mesmos dados)
    X = torch.randn(16, MAX_SEQ_LEN, INPUT_DIM)
    y = torch.randint(0, num_classes, (16,))

    modelo.train()
    for epoch in range(200):
        otimizador.zero_grad()
        loss = criterio(modelo(X), y)
        loss.backward()
        otimizador.step()

    acc = (modelo(X).argmax(1) == y).float().mean().item()
    ok  = acc > 0.95
    log(f'Acurácia no batch após 200 épocas: {acc:.1%}', ok)
    if ok:
        log('Modelo e loop de treino funcionam corretamente.', True)
    else:
        log('FALHA: modelo não consegue memorizar 1 batch. Verifique a arquitetura.', False)
    return ok

# ── TESTE 2 — Sanidade dos dados ─────────────────────────────────────────────

def teste_2_sanidade_dados():
    print(f'\n{SEP}')
    print('TESTE 2 — Sanidade dos dados')
    print(SEP)

    npys = [f for f in os.listdir(DIR_FEATURES) if f.endswith('.npy')]
    log(f'Arquivos .npy (originais): {len(npys)}', len(npys) > 0)

    if os.path.isdir(DIR_AUG):
        n_aug = len([f for f in os.listdir(DIR_AUG) if f.endswith('.npy')])
        log(f'Arquivos .npy (augmentados offline): {n_aug}', n_aug > 0)
    else:
        log('Pasta augmented_features ausente — rode augmentar_offline.py.', None)

    mapa = construir_mapa(DIR_VIDEOS)
    mapeados = sum(1 for f in npys if os.path.splitext(f)[0] in mapa)
    log(f'.npy com classe identificada: {mapeados}/{len(npys)}', mapeados > 0)

    # Verifica shape e conteúdo de 5 arquivos aleatórios
    amostras_ok, zeros_totais = 0, 0
    shapes = set()
    for arq in random.sample(npys, min(20, len(npys))):
        t = np.load(os.path.join(DIR_FEATURES, arq))
        shapes.add(t.shape[1] if t.ndim == 2 else -1)
        frames_com_mao = np.any(t != 0, axis=1).sum() if t.ndim == 2 else 0
        if frames_com_mao == 0:
            zeros_totais += 1
        else:
            amostras_ok += 1

    log(f'Shape dos tensores (coluna): {shapes}  (esperado: {{126}})', shapes == {126})
    log(f'Amostras com mão detectada (amostra de 20): {amostras_ok}/20', amostras_ok >= 15)
    if zeros_totais > 0:
        log(f'Atenção: {zeros_totais} arquivo(s) sem nenhuma detecção de mão.', False)

    # Verifica diversidade entre classes
    amostras, classes = carregar_amostras(max_classes=5)
    if amostras:
        tensores = [paddar(normalizar(np.load(p))) for p, _ in amostras[:5]]
        dists = []
        for i in range(len(tensores)):
            for j in range(i+1, len(tensores)):
                dists.append(np.linalg.norm(tensores[i] - tensores[j]))
        dist_media = np.mean(dists) if dists else 0
        ok_div = dist_media > 1.0
        log(f'Distância média entre amostras de classes diferentes: {dist_media:.3f}', ok_div)
        if not ok_div:
            log('ATENÇÃO: features muito similares entre classes — landmarks podem não estar discriminativos.', False)

    return mapeados > 0 and shapes == {126}

# ── TESTE 3 — Overfit intencional ────────────────────────────────────────────

def teste_3_overfit_intencional():
    print(f'\n{SEP}')
    print('TESTE 3 — Overfit intencional (10 classes, original + augmentadas, sem val)')
    print('  Se o modelo não memoriza essas amostras → features sem informação.')
    print(SEP)

    amostras, classes = carregar_amostras(max_classes=10, max_por_classe=10, incluir_aug=True)
    if len(amostras) < 5:
        log('Amostras insuficientes para este teste.', False)
        return False

    # Remapeia labels para 0..N
    labels_unicos = sorted(set(l for _, l in amostras))
    remap = {old: new for new, old in enumerate(labels_unicos)}
    amostras = [(p, remap[l]) for p, l in amostras]
    num_classes = len(labels_unicos)

    ds     = tensores_para_dataset(amostras)
    loader = DataLoader(ds, batch_size=min(16, len(ds)), shuffle=True)

    modelo     = LIBRASClassifier(input_dim=INPUT_DIM, num_classes=num_classes)
    criterio   = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    for epoch in range(300):
        modelo.train()
        for X, y in loader:
            otimizador.zero_grad()
            criterio(modelo(X), y).backward()
            otimizador.step()

    modelo.eval()
    with torch.no_grad():
        acertos = sum(
            (modelo(X).argmax(1) == y).sum().item()
            for X, y in loader
        )
    acc = acertos / len(ds)
    ok  = acc > 0.70

    log(f'{len(amostras)} amostras, {num_classes} classes reais', True)
    log(f'Acurácia de treino após 300 épocas: {acc:.1%}', ok)
    if ok:
        log('Modelo consegue memorizar dados reais. Problema é de generalização (dados escassos).', True)
    else:
        log('FALHA: modelo não memoriza dados reais. Features podem não ter informação discriminativa.', False)
    return ok

# ── TESTE 4 — Mini-treino com validação ──────────────────────────────────────

def teste_4_mini_treino():
    print(f'\n{SEP}')
    print('TESTE 4 — Mini-treino real (20 classes, original + augmentadas, com validação)')
    print('  Verifica se há qualquer sinal de generalização.')
    print('  OBS: val contém cópias augmentadas do mesmo vídeo (não pessoas novas),')
    print('       então a val_acc aqui é otimista — mede robustez à augmentation.')
    print(SEP)

    amostras, classes = carregar_amostras(max_classes=20, max_por_classe=15, incluir_aug=True)
    if len(amostras) < 10:
        log('Amostras insuficientes.', False)
        return False

    labels_unicos = sorted(set(l for _, l in amostras))
    remap = {old: new for new, old in enumerate(labels_unicos)}
    amostras = [(p, remap[l]) for p, l in amostras]
    num_classes = len(labels_unicos)

    # Split simples: 80/20
    random.shuffle(amostras)
    split     = int(0.8 * len(amostras))
    train_ds  = tensores_para_dataset(amostras[:split])
    val_ds    = tensores_para_dataset(amostras[split:])

    train_loader = DataLoader(train_ds, batch_size=min(16, len(train_ds)), shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=min(16, len(val_ds)),   shuffle=False)

    modelo     = LIBRASClassifier(input_dim=INPUT_DIM, num_classes=num_classes)
    criterio   = nn.CrossEntropyLoss()
    otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

    melhor_val = 0.0
    for epoch in range(100):
        modelo.train()
        for X, y in train_loader:
            otimizador.zero_grad()
            criterio(modelo(X), y).backward()
            otimizador.step()

        if (epoch + 1) % 20 == 0:
            modelo.eval()
            with torch.no_grad():
                acertos = sum((modelo(X).argmax(1) == y).sum().item() for X, y in val_loader)
            val_acc = acertos / len(val_ds) if len(val_ds) > 0 else 0
            melhor_val = max(melhor_val, val_acc)
            print(f'    Epoch {epoch+1:3d}  val_acc={val_acc:.1%}')

    acaso = 1 / num_classes
    ok    = melhor_val > acaso * 1.5
    log(f'Melhor val_acc: {melhor_val:.1%}  |  Acaso: {acaso:.1%}', ok)
    if ok:
        log('Há sinal de generalização. Mais dados e épocas devem melhorar.', True)
    else:
        log('Sem generalização além do acaso. Causa provável: 1 amostra/classe.', False)
    return ok

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 55)
    print(' DIAGNÓSTICO DO PIPELINE — S.I.N.A.I.S')
    print('=' * 55)

    r1 = teste_1_sanidade_modelo()
    r2 = teste_2_sanidade_dados()
    r3 = teste_3_overfit_intencional()
    r4 = teste_4_mini_treino()

    print(f'\n{"=" * 55}')
    print(' RESUMO')
    print('=' * 55)
    resultados = [
        ('Teste 1 — Sanidade do modelo',         r1),
        ('Teste 2 — Sanidade dos dados',          r2),
        ('Teste 3 — Overfit intencional',         r3),
        ('Teste 4 — Generalização mini-treino',   r4),
    ]
    for nome, ok in resultados:
        icone = '✓' if ok else '✗'
        print(f'  [{icone}] {nome}')

    print()
    if r1 and r2 and r3 and not r4:
        print('  DIAGNÓSTICO: Pipeline funciona mas não generaliza, mesmo com augmentation.')
        print('  CAUSA: variações sintéticas do mesmo vídeo não bastam para generalizar.')
        print('  SOLUÇÃO: gravar/obter mais vídeos reais por sinal (1 gravação por palavra é pouco).')
    elif r1 and r2 and not r3:
        print('  DIAGNÓSTICO: Features sem poder discriminativo.')
        print('  CAUSA: Landmarks não capturam diferença entre sinais.')
        print('  SOLUÇÃO: Melhorar extract_features.py (normalização, mais landmarks).')
    elif r1 and not r2:
        print('  DIAGNÓSTICO: Problema nos dados.')
        print('  CAUSA: .npy corrompidos, mapeamento errado ou mão não detectada.')
        print('  SOLUÇÃO: Re-rodar extract_features.py.')
    elif not r1:
        print('  DIAGNÓSTICO: Problema na arquitetura ou loop de treino.')
        print('  SOLUÇÃO: Verificar lstm_classifier.py e train.py.')
    else:
        print('  Todos os testes passaram — sistema funcionando corretamente.')
    print()
