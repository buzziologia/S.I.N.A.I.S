"""
Treina o LIBRASClassifier nos dados extraídos por extract_features.py.

Os tensores gerados têm shape (N_frames, 126) com N variável por vídeo.
O Dataset normaliza cada sequência para MAX_SEQ_LEN frames via padding com zeros
ou truncamento, garantindo batches homogêneos.

Se a pasta data/augmented_features/ existir (gerada por augmentar_offline.py),
ela é combinada com data/processed_features/ para dar múltiplas amostras por classe.
Isso permite divisão train/val/test real com 1 vídeo original por sinal.

Uso:
    python scripts/train.py
    python scripts/train.py --epochs 100 --lr 1e-3 --batch 32
    python scripts/train.py --dir_features data/augmented_features   # só augmentadas
"""
import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Permite importar de models/ a partir da raiz do projeto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.lstm_classifier import LIBRASClassifier

# Comprimento fixo de sequência (frames). Vídeos menores são preenchidos com
# zeros; vídeos maiores são truncados no início (mantém os frames finais do sinal)
MAX_SEQ_LEN = 30
DIM_MAO     = 63                 # 21 landmarks × XYZ (1 mão)
NUM_MAOS    = 2                  # sinais bimanuais → 2 mãos
INPUT_DIM   = DIM_MAO * NUM_MAOS # 126

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def paddar_ou_truncar(tensor: np.ndarray, comprimento: int = MAX_SEQ_LEN) -> np.ndarray:
    """
    Recebe tensor (N, 126) e retorna (comprimento, 126).
    - N > comprimento → mantém os últimos `comprimento` frames
    - N < comprimento → adiciona zeros no início (pre-padding)
    - N == comprimento → retorna sem alteração
    """
    n = len(tensor)
    if n == comprimento:
        return tensor
    if n > comprimento:
        return tensor[-comprimento:]
    pad = np.zeros((comprimento - n, tensor.shape[1]), dtype=np.float32)
    return np.vstack([pad, tensor])


def normalizar_landmarks(tensor: np.ndarray) -> np.ndarray:
    """
    Normaliza os landmarks de cada frame para remover variação de posição e escala.
    Cada mão é normalizada de forma independente (relativa ao seu próprio pulso):
    - Translada ao pulso (landmark 0 = coords [0,1,2] do bloco da mão)
    - Escala pela distância máxima de qualquer ponto ao pulso
    Mãos ausentes (bloco de 63 zeros) são mantidas como zeros.
    """
    out = tensor.copy()
    for i, frame in enumerate(out):
        for h in range(NUM_MAOS):
            ini, fim = h * DIM_MAO, (h + 1) * DIM_MAO
            pts = frame[ini:fim].reshape(21, 3)
            # Pula mãos ausentes
            if np.all(pts == 0):
                continue
            # Translada ao pulso
            pts = pts - pts[0]
            # Escala normalizada
            escala = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
            pts = pts / escala
            out[i, ini:fim] = pts.flatten()
    return out


def _mascara_maos(t: np.ndarray) -> np.ndarray:
    """Retorna máscara (T, NUM_MAOS) booleana: True onde a mão existe (bloco não-nulo)."""
    m = np.zeros((len(t), NUM_MAOS), dtype=bool)
    for h in range(NUM_MAOS):
        bloco = t[:, h * DIM_MAO:(h + 1) * DIM_MAO]
        m[:, h] = ~np.all(bloco == 0, axis=1)
    return m


def aumentar(tensor: np.ndarray, treino: bool = True) -> np.ndarray:
    """
    Augmentation pesada para compensar poucas amostras por palavra.
    Aplica múltiplas transformações aleatórias independentes a cada epoch,
    criando uma versão diferente do mesmo sinal a cada passagem.

    Mãos ausentes (blocos de 63 zeros) são re-zeradas ao final para que o
    ruído/escala não transforme "sem mão" numa mão espúria de escala cheia.
    """
    if not treino:
        return tensor
    t = tensor.copy()
    # Quais mãos existem em cada frame (antes de qualquer ruído).
    # A máscara acompanha os mesmos reordenamentos temporais aplicados a `t`.
    mask = _mascara_maos(t)

    # Ruído gaussiano nos landmarks (simula imprecisão do MediaPipe)
    t += np.random.normal(0, 0.02, t.shape).astype(np.float32)

    # Escala aleatória ±20% (mão mais perto/longe da câmera)
    t *= np.random.uniform(0.80, 1.20)

    # Deslocamento espacial XY em todos os frames (simula posição diferente na tela)
    deslocamento = np.random.uniform(-0.05, 0.05, (1, INPUT_DIM)).astype(np.float32)
    t += deslocamento

    # Velocidade aleatória: reamostrar a sequência com velocidade ±30%
    n = len(t)
    fator = np.random.uniform(0.70, 1.30)
    n_novo = int(np.clip(n * fator, 5, n * 2))
    indices = np.round(np.linspace(0, n - 1, n_novo)).astype(int)
    t    = paddar_ou_truncar(t[indices])               # reajusta para MAX_SEQ_LEN
    mask = paddar_ou_truncar(mask[indices].astype(np.float32)) > 0.5

    # Espelhamento horizontal com 50% de chance: inverte X de cada mão E troca os
    # slots Left/Right (um espelho real troca a mão esquerda pela direita).
    if np.random.rand() < 0.5:
        for i in range(len(t)):
            m0 = t[i, 0:DIM_MAO].reshape(21, 3)
            m1 = t[i, DIM_MAO:INPUT_DIM].reshape(21, 3)
            m0[:, 0] = -m0[:, 0]
            m1[:, 0] = -m1[:, 0]
            t[i, 0:DIM_MAO]         = m1.flatten()
            t[i, DIM_MAO:INPUT_DIM] = m0.flatten()
        mask = mask[:, ::-1].copy()

    # Deslocamento temporal ±4 frames
    shift = np.random.randint(-4, 5)
    if shift != 0:
        t    = np.roll(t, shift, axis=0)
        mask = np.roll(mask, shift, axis=0)

    # Re-zera as mãos que eram ausentes (preserva a semântica "sem mão")
    for h in range(NUM_MAOS):
        t[~mask[:, h], h * DIM_MAO:(h + 1) * DIM_MAO] = 0.0

    return t


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def _construir_mapa_classes(dir_videos: str, assunto: str = None) -> dict:
    """
    Lê a estrutura raw_videos/ASSUNTO/PALAVRA/video.mp4 e retorna
    um dicionário {stem_do_arquivo: palavra}.

    A classe é sempre a PALAVRA — ex: 'AMOR', 'CACHORRO', 'FUTEBOL' —
    para que o modelo aprenda a distinguir sinais individuais.
    Vídeos em 'NENHUM' (sem assunto definido) são incluídos normalmente,
    pois cada palavra ali também é uma classe própria.

    Se `assunto` for informado, só inclui vídeos cujo nome de pasta de
    assunto contém esse texto (case-insensitive). Ex: assunto='FRUTA'.
    """
    filtro = assunto.upper() if assunto else None

    mapa = {}
    for nome_assunto in os.listdir(dir_videos):
        pasta_assunto = os.path.join(dir_videos, nome_assunto)
        if not os.path.isdir(pasta_assunto):
            continue
        # Filtro de assunto (substring case-insensitive)
        if filtro and filtro not in nome_assunto.upper():
            continue
        for palavra in os.listdir(pasta_assunto):
            pasta_palavra = os.path.join(pasta_assunto, palavra)
            if not os.path.isdir(pasta_palavra):
                continue
            for arq in os.listdir(pasta_palavra):
                if arq.endswith('.mp4'):
                    stem = os.path.splitext(arq)[0]
                    mapa[stem] = palavra        # classe = palavra individual
    return mapa


def _carregar_top_k(caminho_csv: str, k: int) -> set:
    """
    Lê as k palavras mais usadas do CSV de frequência (gerado por
    ranquear_palavras.py — coluna 'palavra', já ordenado por frequência).
    """
    import csv
    palavras = []
    with open(caminho_csv, encoding='utf-8') as f:
        for linha in csv.DictReader(f):
            palavras.append(linha['palavra'])
            if len(palavras) >= k:
                break
    return set(palavras)


class LibrasDataset(Dataset):
    def __init__(self, dir_features: str, dir_videos: str = 'data/raw_videos',
                 treino: bool = True, excluir_nenhum: bool = False,
                 assunto: str = None, top_k: int = None,
                 freq_csv: str = 'data/palavras_por_frequencia.csv'):
        self.amostras = []   # [(caminho_npy, label_idx)]
        self.treino   = treino

        mapa_stem_classe = _construir_mapa_classes(dir_videos, assunto=assunto)

        if not mapa_stem_classe:
            raise RuntimeError(f"Nenhum vídeo encontrado em '{dir_videos}'. "
                               "Verifique se os vídeos foram baixados.")

        # Remove a classe NENHUM se solicitado (evita viés de maioria)
        if excluir_nenhum:
            mapa_stem_classe = {s: c for s, c in mapa_stem_classe.items()
                                if c.upper() != 'NENHUM'}
            print("[INFO] Classe NENHUM excluída do treino.")

        # Mantém só as top_k palavras mais usadas (lista de frequência PT-BR)
        if top_k:
            if not os.path.exists(freq_csv):
                raise RuntimeError(f"CSV de frequência não encontrado: {freq_csv}. "
                                   "Rode scripts/ranquear_palavras.py primeiro.")
            top_set = _carregar_top_k(freq_csv, top_k)
            antes = len(set(mapa_stem_classe.values()))
            mapa_stem_classe = {s: c for s, c in mapa_stem_classe.items()
                                if c in top_set}
            print(f"[INFO] top_k={top_k}: {len(set(mapa_stem_classe.values()))} "
                  f"classes mantidas (de {antes}).")

        self.classes      = sorted(set(mapa_stem_classe.values()))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # ── Coleta arquivos: originais + augmentadas offline se disponíveis ───
        # Para arquivos augmentados (stem: nomevideo_aug00) o mapa usa o prefixo
        # antes de "_aug" para encontrar a classe correspondente.
        dir_aug = 'data/augmented_features'
        pastas  = [(dir_features, False)]   # (pasta, é_augmentada)
        if os.path.isdir(dir_aug) and dir_aug != dir_features:
            pastas.append((dir_aug, True))
            print(f"[INFO] Augmentação offline detectada: {dir_aug}")

        sem_classe = 0
        for pasta, e_aug in pastas:
            for arq in os.listdir(pasta):
                if not arq.endswith('.npy'):
                    continue
                stem = os.path.splitext(arq)[0]

                # Para arquivos augmentados (ex: abatidoSm_aug03) usa prefixo
                stem_busca = stem.split('_aug')[0] if e_aug and '_aug' in stem else stem

                classe = mapa_stem_classe.get(stem_busca)
                if classe is None:
                    sem_classe += 1
                    continue
                caminho = os.path.join(pasta, arq)
                self.amostras.append((caminho, self.class_to_idx[classe]))

        if sem_classe:
            print(f"[AVISO] {sem_classe} arquivo(s) sem classe ignorados.")

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, idx):
        caminho, label = self.amostras[idx]
        tensor = np.load(caminho).astype(np.float32)  # (N_frames, 126)
        tensor = paddar_ou_truncar(tensor)             # (MAX_SEQ_LEN, 126)
        tensor = normalizar_landmarks(tensor)          # remove variação posição/escala
        tensor = aumentar(tensor, treino=self.treino)  # augmentation só no treino
        return torch.from_numpy(tensor), label


def _split_estratificado(dataset, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Divide o dataset garantindo que todas as classes apareçam
    no treino, validação e teste (split estratificado).
    """
    from collections import defaultdict
    rng = np.random.default_rng(seed)

    # Agrupa índices por classe
    por_classe = defaultdict(list)
    for i, (_, label) in enumerate(dataset.amostras):
        por_classe[label].append(i)

    train_idx, val_idx, test_idx = [], [], []

    for indices in por_classe.values():
        indices = rng.permutation(indices).tolist()
        n       = len(indices)

        if n == 1:
            # Só 1 amostra: vai apenas para treino
            train_idx += indices
        elif n == 2:
            # 2 amostras: treino + validação
            train_idx += indices[:1]
            val_idx   += indices[1:]
        else:
            n_val   = max(1, int(val_ratio  * n))
            n_test  = max(1, int(test_ratio * n))
            n_train = n - n_val - n_test
            train_idx += indices[:n_train]
            val_idx   += indices[n_train:n_train + n_val]
            test_idx  += indices[n_train + n_val:]

    return train_idx, val_idx, test_idx


# ──────────────────────────────────────────────────────────────────────────────
# Loop de treino / validação
# ──────────────────────────────────────────────────────────────────────────────

def avaliar(modelo, loader, criterio, device):
    modelo.eval()
    total_loss, acertos, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = modelo(X)
            total_loss += criterio(logits, y).item() * len(y)
            acertos += (logits.argmax(1) == y).sum().item()
            total += len(y)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, acertos / total


def _fmt_tempo(segundos: float) -> str:
    """Formata segundos como '12s', '3m 04s' ou '1h 02m'."""
    s = int(max(0, segundos))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def treinar(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    # Dataset completo (augmentation desligada por padrão para splits)
    dataset = LibrasDataset(args.dir_features, args.dir_videos, treino=False,
                            excluir_nenhum=args.excluir_nenhum,
                            assunto=args.assunto, top_k=args.top_k,
                            freq_csv=args.freq_csv)
    if len(dataset) == 0:
        print("[ERRO] Nenhuma amostra encontrada. Execute extract_features.py primeiro.")
        sys.exit(1)

    num_classes = len(dataset.classes)
    print(f"Classes: {num_classes}  |  Amostras: {len(dataset)}")
    # Mostra só uma amostra das classes (a lista inteira polui com milhares de palavras)
    exemplos = ', '.join(dataset.classes[:8])
    print(f"Exemplos de classes: {exemplos}{' ...' if num_classes > 8 else ''}")

    # Split estratificado — garante todas as classes em treino/val/teste
    train_idx, val_idx, test_idx = _split_estratificado(dataset)
    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)

    # Subsets com augmentation apenas no treino
    from torch.utils.data import Subset

    class SubsetComAug(Subset):
        """Subset que ativa augmentation apenas para o split de treino."""
        def __init__(self, dataset, indices, treino):
            super().__init__(dataset, indices)
            self.treino = treino
        def __getitem__(self, idx):
            caminho, label = self.dataset.amostras[self.indices[idx]]
            tensor = np.load(caminho).astype(np.float32)
            tensor = paddar_ou_truncar(tensor)
            tensor = normalizar_landmarks(tensor)
            tensor = aumentar(tensor, treino=self.treino)
            return torch.from_numpy(tensor), label

        def __getitems__(self, indices):
            return [self.__getitem__(idx) for idx in indices]

    train_ds = SubsetComAug(dataset, train_idx, treino=True)
    val_ds   = SubsetComAug(dataset, val_idx,   treino=False)
    test_ds  = SubsetComAug(dataset, test_idx,  treino=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=0)

    # Modelo
    modelo = LIBRASClassifier(input_dim=INPUT_DIM, num_classes=num_classes).to(device)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=args.epochs)

    criterio = nn.CrossEntropyLoss()

    os.makedirs(args.dir_pesos, exist_ok=True)

    # Timestamp (ms) usado como prefixo dos arquivos deste treino.
    # Ex: 1781017673780_modelo.pth / 1781017673780_classes.json
    timestamp     = int(time.time() * 1000)
    caminho_modelo  = os.path.join(args.dir_pesos, f"{timestamp}_modelo.pth")
    caminho_classes = os.path.join(args.dir_pesos, f"{timestamp}_classes.json")
    print(f"[INFO] Treino ID: {timestamp}")

    n_batches = len(train_loader)
    print(f"[INFO] Iniciando treino: {args.epochs} épocas máx | "
          f"{n_train} amostras de treino em {n_batches} batches de {args.batch} | "
          f"{n_val} val | {n_test} teste", flush=True)

    melhor_val_loss  = float('inf')
    melhor_val_acc   = 0.0
    epocas_sem_melhora = 0
    duracoes_epoca = []   # p/ estimar ETA do treino

    # ── Loop principal ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        modelo.train()
        total_loss, acertos, total = 0.0, 0, 0
        inicio_epoca = time.time()
        ult_log = inicio_epoca

        for i, (X, y) in enumerate(train_loader, 1):
            X, y = X.to(device), y.to(device)
            otimizador.zero_grad()
            logits = modelo(X)
            loss = criterio(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()

            total_loss += loss.item() * len(y)
            acertos += (logits.argmax(1) == y).sum().item()
            total += len(y)

            # Progresso dentro da época (no máx. 1 atualização a cada ~5s) —
            # sobrescreve a MESMA linha com '\r' (sem flood) e evita a sensação
            # de "travado" enquanto a primeira época processa tudo.
            agora = time.time()
            if agora - ult_log >= 5.0 or i == n_batches:
                ult_log = agora
                frac      = i / n_batches
                decorrido = agora - inicio_epoca
                eta_ep    = decorrido / frac - decorrido if frac > 0 else 0
                linha = (f"  época {epoch:03d} | batch {i}/{n_batches} ({frac * 100:3.0f}%) | "
                         f"loss {total_loss / total:.4f} | ETA época {_fmt_tempo(eta_ep)}")
                print("\r" + linha.ljust(78), end='', flush=True)

        print()  # finaliza a linha de progresso antes do resumo da época

        scheduler.step()

        train_loss = total_loss / total
        train_acc  = acertos / total
        val_loss, val_acc = avaliar(modelo, val_loader, criterio, device)

        dur_epoca = time.time() - inicio_epoca
        duracoes_epoca.append(dur_epoca)
        media_epoca = sum(duracoes_epoca) / len(duracoes_epoca)
        eta_treino  = media_epoca * (args.epochs - epoch)

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"| {_fmt_tempo(dur_epoca)}/época | ETA treino ≤ {_fmt_tempo(eta_treino)}",
              flush=True)

        # Salva o melhor modelo (monitorado pela val_loss)
        if val_loss < melhor_val_loss:
            melhor_val_loss = val_loss
            melhor_val_acc  = val_acc
            epocas_sem_melhora = 0
            torch.save(modelo.state_dict(), caminho_modelo)
        else:
            epocas_sem_melhora += 1

        # Early stopping
        if epocas_sem_melhora >= args.paciencia:
            print(f"\n[EARLY STOPPING] val_loss não melhorou por {args.paciencia} épocas. "
                  f"Melhor época: {epoch - args.paciencia}  val_loss={melhor_val_loss:.4f}")
            break

    # ── Avaliação final no teste ──────────────────────────────────────────────
    modelo.load_state_dict(torch.load(caminho_modelo))
    test_loss, test_acc = avaliar(modelo, test_loader, criterio, device)
    print(f"\n[RESULTADO FINAL]  test_loss={test_loss:.4f}  test_acc={test_acc:.3f}")

    # Salva mapeamento de classes para uso na inferência
    mapa = {"classes": dataset.classes, "class_to_idx": dataset.class_to_idx}
    with open(caminho_classes, "w", encoding="utf-8") as f:
        json.dump(mapa, f, ensure_ascii=False, indent=2)

    print(f"Pesos salvos em '{caminho_modelo}'")
    print(f"Classes salvas em '{caminho_classes}'")

    imprimir_relatorio(
        num_classes=num_classes,
        n_train=n_train, n_val=n_val, n_test=n_test,
        epochs=args.epochs,
        melhor_val_acc=melhor_val_acc,
        test_acc=test_acc,
        test_loss=test_loss,
        caminho_modelo=caminho_modelo,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Relatório final
# ──────────────────────────────────────────────────────────────────────────────

def imprimir_relatorio(num_classes, n_train, n_val, n_test, epochs,
                       melhor_val_acc, test_acc, test_loss, caminho_modelo):
    """Imprime um resumo legível do treino no terminal."""

    # Avaliação qualitativa da acurácia
    if test_acc >= 0.85:
        avaliacao = "Ótimo ✓"
    elif test_acc >= 0.70:
        avaliacao = "Bom"
    elif test_acc >= 0.50:
        avaliacao = "Razoável — considere mais dados ou mais épocas"
    else:
        avaliacao = "Fraco — verifique os dados e o mapeamento de classes"

    # Diagnóstico de overfitting
    gap = melhor_val_acc - test_acc
    if melhor_val_acc > 0 and (melhor_val_acc - test_acc) > 0.15:
        diagnostico = "Possível overfitting (val_acc muito maior que test_acc)"
    elif test_acc < 1 / num_classes + 0.05:
        diagnostico = "Acurácia próxima ao acaso — modelo pode não estar aprendendo"
    else:
        diagnostico = "Sem sinais evidentes de overfitting"

    sep = "═" * 52
    print(f"\n{sep}")
    print(f"  RELATÓRIO DE TREINO — S.I.N.A.I.S")
    print(sep)
    print(f"  Classes treinadas   : {num_classes}")
    print(f"  Amostras  treino    : {n_train}")
    print(f"  Amostras  validação : {n_val}")
    print(f"  Amostras  teste     : {n_test}")
    print(f"  Épocas              : {epochs}")
    print(f"{sep}")
    print(f"  Melhor val_acc      : {melhor_val_acc * 100:.1f}%")
    print(f"  Acurácia no teste   : {test_acc * 100:.1f}%")
    print(f"  Loss no teste       : {test_loss:.4f}")
    print(f"  Acaso esperado      : {100 / num_classes:.1f}%  (1 / {num_classes} classes)")
    print(f"{sep}")
    print(f"  Avaliação           : {avaliacao}")
    print(f"  Diagnóstico         : {diagnostico}")
    print(f"  Pesos salvos em     : {caminho_modelo}")
    print(f"{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina o LIBRASClassifier")
    parser.add_argument("--dir_features", default="data/processed_features",
                        help="Pasta com os .npy extraídos")
    parser.add_argument("--dir_videos",   default="data/raw_videos",
                        help="Pasta com os vídeos originais (para mapear classes)")
    parser.add_argument("--dir_pesos",    default="models/saved_weights",
                        help="Onde salvar os pesos treinados")
    parser.add_argument("--epochs",   type=int,   default=200,  help="Número máximo de épocas")
    parser.add_argument("--batch",    type=int,   default=16,   help="Tamanho do batch")
    parser.add_argument("--lr",       type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--paciencia",type=int,   default=15,   help="Early stopping: épocas sem melhora na val_loss")
    parser.add_argument("--excluir_nenhum", action="store_true",
                        help="Exclui as palavras do assunto NENHUM do treino")
    parser.add_argument("--assunto",        default=None,
                        help="Treinar só um assunto (ex: FRUTA). Sem isso = todas as palavras")
    parser.add_argument("--top_k",          type=int, default=None,
                        help="Treinar só nas K palavras mais usadas (lista de frequência PT-BR)")
    parser.add_argument("--freq_csv",       default="data/palavras_por_frequencia.csv",
                        help="CSV de frequência gerado por ranquear_palavras.py")
    args = parser.parse_args()

    treinar(args)
