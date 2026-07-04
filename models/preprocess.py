"""
Pré-processamento compartilhado de sequências de landmarks (N, 126).

Fonte ÚNICA das transformações usadas em treino, avaliação e inferência ao
vivo (train.py, avaliar_holdout.py, testar_camera.py). Qualquer mudança de
convenção — janela, recorte, normalização — deve acontecer AQUI, para que
treino e inferência nunca divirjam.

Convenção do vetor de frame (126,):
    slot 0 (0:63)   = mão rotulada 'Left'  pelo MediaPipe
    slot 1 (63:126) = mão rotulada 'Right'
    mão ausente     = bloco de 63 zeros

Todas as funções são vetorizadas em numpy (sem loop Python por frame) —
são executadas a cada __getitem__ do treino, então desempenho importa.
"""
import re

import numpy as np

MAX_SEQ_LEN = 30                 # comprimento fixo de sequência (frames)
DIM_MAO     = 63                 # 21 landmarks × XYZ (1 mão)
NUM_MAOS    = 2                  # sinais bimanuais → 2 mãos
INPUT_DIM   = DIM_MAO * NUM_MAOS # 126


def paddar_ou_truncar(tensor: np.ndarray, comprimento: int = MAX_SEQ_LEN) -> np.ndarray:
    """
    Recebe tensor (N, D) e retorna (comprimento, D).
    - N > comprimento → mantém os últimos `comprimento` frames
    - N < comprimento → adiciona zeros no início (pre-padding)
    """
    n = len(tensor)
    if n == comprimento:
        return tensor
    if n > comprimento:
        return tensor[-comprimento:]
    pad = np.zeros((comprimento - n, tensor.shape[1]), dtype=np.float32)
    return np.vstack([pad, tensor])


def recortar_atividade(tensor: np.ndarray, margem: int = 2) -> np.ndarray:
    """
    Corta a sequência para a janela onde alguma mão foi detectada (± margem).
    Vídeos coletados pela webcam têm o sinal no MEIO do take (contagem antes,
    braço abaixando depois) — sem este recorte, manter os últimos 30 frames
    pode descartar o sinal inteiro e entregar um tensor todo de zeros.
    Nos vídeos INES o sinal ocupa o clipe quase todo, então o recorte é neutro.
    """
    mao = ~np.all(tensor == 0, axis=1)
    idx = np.where(mao)[0]
    if len(idx) == 0:
        return tensor
    ini = max(0, idx[0] - margem)
    fim = min(len(tensor), idx[-1] + margem + 1)
    return tensor[ini:fim]


def mascara_maos(tensor: np.ndarray) -> np.ndarray:
    """Retorna máscara (T, NUM_MAOS) booleana: True onde a mão existe (bloco não-nulo)."""
    blocos = tensor.reshape(len(tensor), NUM_MAOS, DIM_MAO)
    return ~np.all(blocos == 0, axis=2)


def normalizar_landmarks(tensor: np.ndarray) -> np.ndarray:
    """
    Normaliza os landmarks de cada frame para remover variação de posição e
    escala. Cada mão é normalizada de forma independente:
    - Translada ao pulso (landmark 0 da mão)
    - Escala pela distância máxima de qualquer ponto ao pulso
    Mãos ausentes (bloco de 63 zeros) são mantidas como zeros.
    """
    t = np.ascontiguousarray(tensor, dtype=np.float32)
    pts = t.reshape(len(t), NUM_MAOS, 21, 3).copy()      # (T, 2, 21, 3)
    presente = ~np.all(pts.reshape(len(t), NUM_MAOS, DIM_MAO) == 0, axis=2)

    pts -= pts[:, :, 0:1, :]                              # translada ao pulso
    escala = np.linalg.norm(pts, axis=3).max(axis=2)      # (T, 2)
    pts /= (escala + 1e-6)[:, :, None, None]
    pts[~presente] = 0.0                                  # preserva "sem mão"
    return pts.reshape(len(t), INPUT_DIM)


def preparar_sequencia(tensor: np.ndarray) -> np.ndarray:
    """
    Pipeline padrão de inferência: recorta a janela de atividade, ajusta para
    MAX_SEQ_LEN frames e normaliza. É EXATAMENTE o que o treino aplica antes
    da augmentation — use isto em qualquer avaliação ou inferência.
    """
    t = np.asarray(tensor, dtype=np.float32)
    return normalizar_landmarks(paddar_ou_truncar(recortar_atividade(t)))


# ──────────────────────────────────────────────────────────────────────────────
# Unificação de variantes NA CLASSIFICAÇÃO (não no treino)
#
# O dicionário INES tem execuções alternativas do mesmo sinal como palavras
# numeradas (QUE1/QUE2, NÃO1/NÃO2...). O modelo é treinado com as variantes
# separadas, mas na hora de CLASSIFICAR as probabilidades das variantes são
# somadas e o resultado é exibido pela palavra-base ('QUE'): o usuário não
# se importa qual variante executou, só qual palavra é.
# ──────────────────────────────────────────────────────────────────────────────

def colapsar_variante(nome: str, bases: set) -> str:
    """'QUE1' → 'QUE' se 'QUE' estiver em bases; caso contrário, nome original."""
    m = re.fullmatch(r'(.+?)\d+', nome)
    if m and m.group(1).upper() in bases:
        return m.group(1)
    return nome


def parse_bases(spec: str) -> set:
    """Converte '--unificar QUE,NÃO' num set de bases em maiúsculas ('' = nenhum)."""
    return {b.strip().upper() for b in (spec or '').split(',') if b.strip()}


def unificar_rotulos(classes: list, bases: set):
    """
    Prepara a fusão de variantes na saída do modelo.
    Retorna (rotulos, mapa_idx):
      - rotulos:  lista de rótulos exibidos (variantes das bases colapsadas)
      - mapa_idx: array (n_classes,) com o índice do rótulo de cada classe
    Para agrupar probabilidades: np.add.at(probs_u, mapa_idx, probs).
    """
    exibidos = [colapsar_variante(c, bases) for c in classes]
    rotulos = list(dict.fromkeys(exibidos))          # únicos, na ordem original
    idx = {r: i for i, r in enumerate(rotulos)}
    return rotulos, np.array([idx[e] for e in exibidos])


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
    mask = mascara_maos(t)

    # Ruído gaussiano nos landmarks (simula imprecisão do MediaPipe)
    t += np.random.normal(0, 0.02, t.shape).astype(np.float32)

    # Escala aleatória ±20% (mão mais perto/longe da câmera)
    t *= np.random.uniform(0.80, 1.20)

    # Deslocamento espacial XY em todos os frames (simula posição diferente na tela)
    t += np.random.uniform(-0.05, 0.05, (1, INPUT_DIM)).astype(np.float32)

    # Velocidade aleatória: reamostrar a sequência com velocidade ±30%
    n = len(t)
    fator = np.random.uniform(0.70, 1.30)
    n_novo = int(np.clip(n * fator, 5, n * 2))
    indices = np.round(np.linspace(0, n - 1, n_novo)).astype(int)
    t    = paddar_ou_truncar(t[indices])               # reajusta para MAX_SEQ_LEN
    mask = paddar_ou_truncar(mask[indices].astype(np.float32)) > 0.5

    # Espelhamento horizontal com 50% de chance: inverte X de cada mão E troca
    # os slots Left/Right (um espelho real troca a mão esquerda pela direita).
    if np.random.rand() < 0.5:
        v = t.reshape(len(t), NUM_MAOS, 21, 3).copy()
        v[..., 0] = -v[..., 0]
        t = v[:, ::-1].reshape(len(t), INPUT_DIM)
        mask = mask[:, ::-1].copy()

    # Deslocamento temporal ±4 frames
    shift = np.random.randint(-4, 5)
    if shift != 0:
        t    = np.roll(t, shift, axis=0)
        mask = np.roll(mask, shift, axis=0)

    # Re-zera as mãos que eram ausentes (preserva a semântica "sem mão")
    t = np.ascontiguousarray(t)
    t.reshape(len(t), NUM_MAOS, DIM_MAO)[~mask] = 0.0

    return t
