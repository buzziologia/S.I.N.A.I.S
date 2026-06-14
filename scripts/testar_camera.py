"""
Teste em tempo real — captura a câmera, extrai landmarks com MediaPipe,
alimenta o modelo LSTM e exibe o sinal identificado na tela.

Uso:
    python scripts/testar_camera.py
    python scripts/testar_camera.py --pesos models/saved_weights/melhor_modelo.pth
    python scripts/testar_camera.py --confianca 0.6
"""
import argparse
import collections
import json
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from models.lstm_classifier import LIBRASClassifier
import train  # reusa o pré-processamento idêntico ao treino p/ montar o banco OOD

# ──────────────────────────────────────────────────────────────────────────────
# Configurações
# ──────────────────────────────────────────────────────────────────────────────
MAX_SEQ_LEN = 30                  # deve ser igual ao usado no treino
DIM_MAO     = 63                  # 21 landmarks × XYZ (1 mão)
NUM_MAOS    = 2                   # sinais bimanuais → 2 mãos
INPUT_DIM   = DIM_MAO * NUM_MAOS  # 126

mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (mesma lógica do train.py)
# ──────────────────────────────────────────────────────────────────────────────

def normalizar_landmarks(frame: np.ndarray) -> np.ndarray:
    """
    Normaliza 1 frame (126,): cada mão é normalizada de forma independente
    (translada ao próprio pulso e escala pelo alcance máximo).
    Mãos ausentes (bloco de 63 zeros) permanecem zeradas.
    """
    out = frame.copy()
    for h in range(NUM_MAOS):
        ini, fim = h * DIM_MAO, (h + 1) * DIM_MAO
        pts = out[ini:fim].reshape(21, 3)
        if np.all(pts == 0):
            continue
        pts = pts - pts[0]
        escala = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
        out[ini:fim] = (pts / escala).flatten()
    return out


def paddar_sequencia(buffer: list) -> np.ndarray:
    """Converte o buffer de frames em tensor (MAX_SEQ_LEN, 126) com padding."""
    arr = np.array(buffer, dtype=np.float32)
    n   = len(arr)
    if n >= MAX_SEQ_LEN:
        return arr[-MAX_SEQ_LEN:]
    pad = np.zeros((MAX_SEQ_LEN - n, INPUT_DIM), dtype=np.float32)
    return np.vstack([pad, arr])


def extrair_landmarks(results) -> np.ndarray:
    """
    Extrai vetor (126,) das duas mãos de um frame processado pelo MediaPipe.
    A mão 'Left' ocupa o bloco 0:63 e a 'Right' o bloco 63:126 — mesma regra do
    extract_features.py. Mão ausente → bloco de zeros.
    """
    vetor = np.zeros(INPUT_DIM, dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for mao_lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            rotulo = handed.classification[0].label   # 'Left' ou 'Right'
            base   = (0 if rotulo == 'Left' else 1) * DIM_MAO
            pts = []
            for lm in mao_lm.landmark:
                pts.extend([lm.x, lm.y, lm.z])
            vetor[base:base + DIM_MAO] = np.array(pts, dtype=np.float32)
    return normalizar_landmarks(vetor)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def encontrar_modelo_mais_recente(dir_pesos: str) -> str:
    """Retorna o caminho do *_modelo.pth mais recente na pasta de pesos."""
    import glob
    candidatos = glob.glob(os.path.join(dir_pesos, "*_modelo.pth"))
    # Compatibilidade com o nome antigo
    legado = os.path.join(dir_pesos, "melhor_modelo.pth")
    if os.path.exists(legado):
        candidatos.append(legado)
    if not candidatos:
        return None
    return max(candidatos, key=os.path.getmtime)


def derivar_caminho_classes(caminho_pesos: str) -> str:
    """
    Deriva o arquivo de classes a partir do nome dos pesos.
    Ex: '1781017673780_modelo.pth' → '1781017673780_classes.json'
    Compatível com o nome antigo 'classes.json'.
    """
    base = os.path.basename(caminho_pesos)
    pasta = os.path.dirname(caminho_pesos)
    if base.endswith("_modelo.pth"):
        candidato = os.path.join(pasta, base.replace("_modelo.pth", "_classes.json"))
        if os.path.exists(candidato):
            return candidato
    # Fallback: classes.json na mesma pasta (formato legado)
    return os.path.join(pasta, 'classes.json')


# ──────────────────────────────────────────────────────────────────────────────
# Detecção de OOD (KNN sobre as features penúltimas)
# ──────────────────────────────────────────────────────────────────────────────

def extrair_feature(modelo, x):
    """Roda o modelo e devolve (logits, feature_penultima) numa só passada."""
    out, _ = modelo.lstm(x)
    last   = out[:, -1, :]
    feat   = modelo.classifier[:5](last)        # penúltima camada (antes do Linear final)
    logits = modelo.classifier[5](feat)
    return logits[0], feat[0]


def construir_banco_ood(modelo, classes, device, dir_features='data/processed_features',
                        dir_videos='data/raw_videos', k=5, percentil=95):
    """
    Monta um banco de features das amostras de TREINO das classes do modelo e
    calibra um limiar de distância KNN (percentil) para rejeitar entradas OOD.
    Retorna dict {banco, k, limiar} ou None se não der pra calibrar.
    """
    try:
        mapa = train._construir_mapa_classes(dir_videos)   # {stem: palavra}
    except Exception as e:
        print(f"[OOD] não consegui mapear classes ({e}). OOD desativado.")
        return None

    classes_set = set(classes)
    feats = []
    modelo.eval()
    with torch.no_grad():
        for stem, palavra in mapa.items():
            if palavra not in classes_set:
                continue
            p = os.path.join(dir_features, stem + '.npy')
            if not os.path.exists(p):
                continue
            t = train.normalizar_landmarks(train.paddar_ou_truncar(
                np.load(p).astype(np.float32)))
            x = torch.from_numpy(t).unsqueeze(0).to(device)
            _, feat = extrair_feature(modelo, x)
            feats.append(feat.cpu().numpy())

    if len(feats) < max(10, k + 1):
        print(f"[OOD] poucas features ({len(feats)}) p/ calibrar. OOD desativado.")
        return None

    banco = np.array(feats, dtype=np.float32)
    banco /= (np.linalg.norm(banco, axis=1, keepdims=True) + 1e-8)
    # distância de cada amostra ao k-ésimo vizinho (excluindo ela mesma)
    a2 = (banco ** 2).sum(1)
    D  = np.sqrt(np.maximum(a2[:, None] + a2[None, :] - 2 * banco @ banco.T, 0))
    np.fill_diagonal(D, np.inf)
    D.sort(axis=1)
    limiar = float(np.percentile(D[:, k - 1], percentil))
    print(f"[OOD] banco com {len(banco)} features | limiar KNN (k={k}, p{percentil}) = {limiar:.3f}")
    return {'banco': banco, 'k': k, 'limiar': limiar}


def score_ood(feat, ood):
    """Distância ao k-ésimo vizinho no banco (maior = mais OOD)."""
    f = feat / (np.linalg.norm(feat) + 1e-8)
    d = np.sqrt(np.maximum(((ood['banco'] - f) ** 2).sum(1), 0))
    d.sort()
    return float(d[min(ood['k'], len(d)) - 1])


def mapear_videos(classes, dir_videos='data/raw_videos'):
    """Retorna {palavra: caminho_do_mp4} para as palavras do modelo (1º vídeo achado)."""
    alvo = set(classes)
    mapa = {}
    if not os.path.isdir(dir_videos):
        return mapa
    for assunto in os.listdir(dir_videos):
        pasta_a = os.path.join(dir_videos, assunto)
        if not os.path.isdir(pasta_a):
            continue
        for palavra in os.listdir(pasta_a):
            if palavra not in alvo or palavra in mapa:
                continue
            pasta_p = os.path.join(pasta_a, palavra)
            if not os.path.isdir(pasta_p):
                continue
            mp4s = [f for f in os.listdir(pasta_p) if f.lower().endswith('.mp4')]
            if mp4s:
                mapa[palavra] = os.path.join(pasta_p, mp4s[0])
    return mapa


def main(args):
    # ── Resolve o caminho dos pesos ────────────────────────────────────────────
    if args.pesos is None:
        args.pesos = encontrar_modelo_mais_recente(args.dir_pesos)
        if args.pesos is None:
            print(f"[ERRO] Nenhum modelo (*_modelo.pth) encontrado em '{args.dir_pesos}'. "
                  "Treine o modelo primeiro.")
            sys.exit(1)
        print(f"[INFO] Usando modelo mais recente: {args.pesos}")

    if not os.path.exists(args.pesos):
        print(f"[ERRO] Pesos não encontrados: {args.pesos}")
        sys.exit(1)

    # ── Carrega classes (mesmo timestamp dos pesos) ────────────────────────────
    classes_path = derivar_caminho_classes(args.pesos)
    if not os.path.exists(classes_path):
        print(f"[ERRO] Arquivo de classes não encontrado: {classes_path}. "
              "Treine o modelo primeiro.")
        sys.exit(1)
    print(f"[INFO] Classes: {classes_path}")

    # Limiar de confiança: usa --confianca se informado, senão 0.6
    conf_threshold = args.confianca if args.confianca is not None else 0.6
    print(f"[INFO] Limiar de confiança: {conf_threshold}")

    with open(classes_path, encoding='utf-8') as f:
        dados = json.load(f)
    classes = dados['classes']
    num_classes = len(classes)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Infere dimensões do checkpoint para não depender da definição atual do modelo
    state = torch.load(args.pesos, map_location=device)
    hidden_dim  = state['lstm.weight_hh_l0'].shape[1]          # hidden size
    intermediario = state['classifier.2.weight'].shape[0]       # camada intermediária
    num_classes_ckpt = state['classifier.5.weight'].shape[0]    # saída

    if num_classes_ckpt != num_classes:
        print(f"[AVISO] Checkpoint tem {num_classes_ckpt} classes, "
              f"classes.json tem {num_classes}. Usando o checkpoint.")
        num_classes = num_classes_ckpt

    modelo = LIBRASClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)

    # Ajusta camada intermediária se diferente do padrão
    if intermediario != 256:
        import torch.nn as nn
        modelo.classifier[2] = nn.Linear(hidden_dim * 2, intermediario)
        modelo.classifier[5] = nn.Linear(intermediario, num_classes)
        modelo.to(device)

    modelo.load_state_dict(state)
    modelo.eval()
    print(f"Modelo carregado — {num_classes} classes | hidden={hidden_dim} | dispositivo: {device}")

    # ── Banco de OOD (rejeita sinais fora do vocabulário treinado) ─────────────
    ood = None
    if not args.sem_ood:
        ood = construir_banco_ood(modelo, classes, device,
                                  k=args.k_ood, percentil=args.ood_percentil)
        if ood is None:
            print("[OOD] desativado (sem banco). Mostrando todas as predições.")

    # ── Câmera e MediaPipe ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Não foi possível abrir a câmera.")
        sys.exit(1)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    buffer_frames  = collections.deque(maxlen=MAX_SEQ_LEN)
    sinal_atual    = "—"
    confianca_atual = 0.0
    status         = ""   # motivo de não exibir sinal (sem mão / fora do vocabulário)
    # Suaviza predições: mantém a última predição por N frames para evitar flicker
    historico_pred = collections.deque(maxlen=8)

    # Vídeo de referência da palavra identificada (canto inferior direito).
    palavra_para_video = mapear_videos(classes)
    print(f"[VÍDEO] {len(palavra_para_video)} palavras com vídeo de referência.")
    video_cap  = None   # VideoCapture do vídeo tocando agora (None = nenhum)
    video_word = None   # palavra do vídeo tocando

    print("Câmera aberta. Pressione 'Q' para sair.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)   # espelha horizontalmente (mais natural)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Extração de landmarks ─────────────────────────────────────────────
        results = hands.process(rgb)
        vetor   = extrair_landmarks(results)
        buffer_frames.append(vetor)

        # ── Desenha landmarks na mão ──────────────────────────────────────────
        if results.multi_hand_landmarks:
            for mao_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, mao_lm, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

        # ── Inferência (só quando o buffer está cheio) ────────────────────────
        if len(buffer_frames) == MAX_SEQ_LEN:
            # Gate 1: presença de mão (fração de frames do buffer com mão detectada)
            frac_maos = float(np.mean([np.any(f != 0) for f in buffer_frames]))

            tensor = paddar_sequencia(list(buffer_frames))
            x      = torch.from_numpy(tensor).unsqueeze(0).to(device)  # (1, 30, 126)

            with torch.no_grad():
                logits, feat = extrair_feature(modelo, x)
                probs = torch.softmax(logits, dim=0)
                idx   = probs.argmax().item()
                conf  = probs[idx].item()

            # Gate 2: OOD por KNN (entrada longe das amostras de treino)
            eh_ood = False
            if ood is not None:
                eh_ood = score_ood(feat.cpu().numpy(), ood) > ood['limiar']

            if frac_maos < args.min_maos:
                status = "sem mao"
                historico_pred.clear()
                sinal_atual, confianca_atual = "—", 0.0
            elif eh_ood:
                status = "fora do vocabulario"
                historico_pred.clear()
                sinal_atual, confianca_atual = "—", 0.0
            else:
                status = ""
                if conf >= conf_threshold:
                    historico_pred.append(idx)
                # Predição mais frequente no histórico (suavização)
                if historico_pred:
                    pred_suave      = max(set(historico_pred), key=historico_pred.count)
                    sinal_atual     = classes[pred_suave]
                    confianca_atual = conf

        # ── Interface visual ──────────────────────────────────────────────────
        # Fundo semitransparente no topo
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # Nome do sinal
        cv2.putText(frame, sinal_atual, (15, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 120), 3, cv2.LINE_AA)

        # Confiança
        conf_txt = f"{confianca_atual * 100:.0f}%"
        cv2.putText(frame, conf_txt, (w - 80, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)

        # Barra de confiança
        barra_w = int((w - 20) * confianca_atual)
        cv2.rectangle(frame, (10, 58), (10 + barra_w, 66), (0, 255, 120), -1)
        cv2.rectangle(frame, (10, 58), (w - 10, 66), (100, 100, 100), 1)

        # Status de rejeição (sem mão / fora do vocabulário)
        if status:
            cv2.putText(frame, status, (15, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 180, 255), 2, cv2.LINE_AA)

        # ── Vídeo de referência da palavra (canto inferior direito) ───────────
        # Só inicia um vídeo novo quando NÃO há outro tocando (não troca no meio).
        if video_cap is None and sinal_atual in palavra_para_video:
            video_cap  = cv2.VideoCapture(palavra_para_video[sinal_atual])
            video_word = sinal_atual

        if video_cap is not None:
            ok_v, vframe = video_cap.read()
            if not ok_v:                      # vídeo terminou → libera p/ o próximo
                video_cap.release()
                video_cap = None
            else:
                # miniatura no canto inferior direito (vídeo cru, sem landmarks)
                tw = w // 4
                th = int(vframe.shape[0] * tw / vframe.shape[1])
                th = min(th, h - 40)
                thumb = cv2.resize(vframe, (tw, th))
                x1, y1 = w - tw - 10, h - th - 10
                frame[y1:y1 + th, x1:x1 + tw] = thumb
                cv2.rectangle(frame, (x1, y1), (x1 + tw, y1 + th), (0, 255, 120), 2)
                cv2.putText(frame, video_word, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2, cv2.LINE_AA)

        # Instrução no rodapé
        cv2.putText(frame, "Q: sair", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("S.I.N.A.I.S — Reconhecimento de Libras", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    hands.close()
    if video_cap is not None:
        video_cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconhecimento de Libras em tempo real')
    parser.add_argument('--pesos',     default=None,
                        help='Caminho para os pesos (ex: models/saved_weights/1781017673780_modelo.pth). '
                             'Sem isso, usa o modelo mais recente.')
    parser.add_argument('--dir_pesos', default='models/saved_weights',
                        help='Pasta onde procurar o modelo mais recente quando --pesos não é informado')
    parser.add_argument('--confianca', type=float, default=None,
                        help='Confiança mínima para exibir predição (0.0–1.0). Padrão: 0.6')
    parser.add_argument('--sem_ood', action='store_true',
                        help='Desativa a detecção de OOD (mostra todas as predições)')
    parser.add_argument('--ood_percentil', type=float, default=95,
                        help='Percentil das distâncias de treino p/ o limiar de OOD. '
                             'Menor = mais rígido (rejeita mais). Padrão: 95')
    parser.add_argument('--k_ood', type=int, default=5,
                        help='k do KNN usado na detecção de OOD. Padrão: 5')
    parser.add_argument('--min_maos', type=float, default=0.5,
                        help='Fração mínima de frames com mão no buffer p/ tentar prever. Padrão: 0.5')
    main(parser.parse_args())
