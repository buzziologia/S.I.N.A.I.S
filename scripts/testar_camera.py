"""
Teste em tempo real - captura a câmera, extrai landmarks com MediaPipe,
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
# Pré-processamento IDÊNTICO ao do treino (fonte única em models/preprocess.py):
# recorta a janela de atividade da mão, padda/trunca para MAX_SEQ_LEN e normaliza.
from models.preprocess import (MAX_SEQ_LEN, DIM_MAO, INPUT_DIM, preparar_sequencia,
                               colapsar_variante, parse_bases, unificar_rotulos)
import train  # p/ mapear classes (stem → palavra) ao montar o banco OOD

mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils


def extrair_landmarks(results) -> np.ndarray:
    """
    Extrai vetor CRU (126,) das duas mãos de um frame processado pelo MediaPipe.
    A mão 'Left' ocupa o bloco 0:63 e a 'Right' o bloco 63:126 - mesma regra do
    extract_features.py. Mão ausente → bloco de zeros.
    A normalização acontece depois, na janela inteira (preparar_sequencia),
    exatamente como no treino.
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
    return vetor


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


def construir_banco_ood(modelo, classes, device, caminho_pesos,
                        dir_features='data/processed_features',
                        dir_videos='data/raw_videos', k=5, percentil=95):
    """
    Monta um banco de features das amostras de TREINO das classes do modelo e
    calibra um limiar de distância KNN (percentil) para rejeitar entradas OOD.
    O banco é cacheado em disco ao lado dos pesos (*_ood_kK_pP.npz) - montar do
    zero exige uma passada do modelo por TODAS as amostras, o que é lento em
    vocabulários grandes. Retorna dict {banco, k, limiar} ou None.
    """
    cache_path = None
    if caminho_pesos.endswith('_modelo.pth'):
        cache_path = caminho_pesos.replace(
            '_modelo.pth', f'_ood_k{k}_p{int(percentil)}.npz')
        if os.path.exists(cache_path):
            dados = np.load(cache_path)
            print(f"[OOD] banco carregado do cache: {cache_path} "
                  f"({len(dados['banco'])} features | limiar = {float(dados['limiar']):.3f})")
            return {'banco': dados['banco'], 'k': k, 'limiar': float(dados['limiar'])}

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
            t = preparar_sequencia(np.load(p))
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

    if cache_path:
        np.savez(cache_path, banco=banco, limiar=limiar)
        print(f"[OOD] banco salvo em cache: {cache_path}")
    return {'banco': banco, 'k': k, 'limiar': limiar}


def score_ood(feat, ood):
    """Distância ao k-ésimo vizinho no banco (maior = mais OOD)."""
    f = feat / (np.linalg.norm(feat) + 1e-8)
    d = np.sqrt(np.maximum(((ood['banco'] - f) ** 2).sum(1), 0))
    d.sort()
    return float(d[min(ood['k'], len(d)) - 1])


def mapear_videos(rotulos, bases, dir_videos='data/raw_videos'):
    """
    Retorna {rótulo_exibido: caminho_do_mp4} (1º vídeo achado). As pastas de
    variantes (QUE1/QUE2) atendem pelo rótulo unificado ('QUE').
    """
    alvo = set(rotulos)
    mapa = {}
    if not os.path.isdir(dir_videos):
        return mapa
    for assunto in os.listdir(dir_videos):
        pasta_a = os.path.join(dir_videos, assunto)
        if not os.path.isdir(pasta_a):
            continue
        for palavra in os.listdir(pasta_a):
            rotulo = colapsar_variante(palavra, bases)
            if rotulo not in alvo or rotulo in mapa:
                continue
            pasta_p = os.path.join(pasta_a, palavra)
            if not os.path.isdir(pasta_p):
                continue
            mp4s = [f for f in os.listdir(pasta_p) if f.lower().endswith('.mp4')]
            if mp4s:
                mapa[rotulo] = os.path.join(pasta_p, mp4s[0])
    return mapa


def texto_caixa(frame, txt, org, escala=0.7, cor=(255, 255, 255), thick=2):
    """Desenha texto com uma caixa preta atrás - legível sobre qualquer fundo."""
    (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, escala, thick)
    x, y = org
    pad = 7
    cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + base + pad),
                  (0, 0, 0), -1)
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, thick, cv2.LINE_AA)


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
    print(f"Modelo carregado - {num_classes} classes | hidden={hidden_dim} | dispositivo: {device}")

    # Fusão de variantes na CLASSIFICAÇÃO: o modelo continua prevendo QUE1/QUE2,
    # mas as probabilidades são somadas e o rótulo exibido é a palavra-base.
    bases = parse_bases(args.unificar)
    rotulos, mapa_idx = unificar_rotulos(classes, bases)
    if len(rotulos) < len(classes):
        print(f"[INFO] Variantes unificadas na exibição ({args.unificar}): "
              f"{len(classes)} classes → {len(rotulos)} rótulos.")

    # ── Banco de OOD (rejeita sinais fora do vocabulário treinado) ─────────────
    ood = None
    if not args.sem_ood:
        ood = construir_banco_ood(modelo, classes, device, args.pesos,
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
    sinal_atual    = "-"
    confianca_atual = 0.0
    status         = ""   # motivo de não exibir sinal (sem mão / fora do vocabulário)
    # Suaviza predições: mantém a última predição por N frames para evitar flicker
    historico_pred = collections.deque(maxlen=8)

    # Vídeo de referência da palavra identificada (canto inferior direito).
    palavra_para_video = mapear_videos(rotulos, bases)
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
            arr = np.array(buffer_frames, dtype=np.float32)   # (30, 126) cru

            # Gate 1: presença de mão (fração de frames do buffer com mão detectada)
            frac_maos = float(np.mean(np.any(arr != 0, axis=1)))

            # Mesmo pipeline do treino: recorta a janela com mão, padda e normaliza
            tensor = preparar_sequencia(arr)
            x      = torch.from_numpy(tensor).unsqueeze(0).to(device)  # (1, 30, 126)

            with torch.no_grad():
                logits, feat = extrair_feature(modelo, x)
                probs = torch.softmax(logits, dim=0).cpu().numpy()

            # Soma as probabilidades das variantes unificadas (QUE1+QUE2 → QUE)
            probs_u = np.zeros(len(rotulos))
            np.add.at(probs_u, mapa_idx, probs)
            idx  = int(probs_u.argmax())
            conf = float(probs_u[idx])

            # Gate 2: OOD por KNN (entrada longe das amostras de treino)
            eh_ood = False
            if ood is not None:
                eh_ood = score_ood(feat.cpu().numpy(), ood) > ood['limiar']

            if frac_maos < args.min_maos:
                status = "sem mao"
                historico_pred.clear()
                sinal_atual, confianca_atual = "-", 0.0
            elif eh_ood:
                status = "fora do vocabulario"
                historico_pred.clear()
                sinal_atual, confianca_atual = "-", 0.0
            else:
                status = ""
                if conf >= conf_threshold:
                    historico_pred.append(idx)
                # Predição mais frequente no histórico (suavização)
                if historico_pred:
                    pred_suave      = max(set(historico_pred), key=historico_pred.count)
                    sinal_atual     = rotulos[pred_suave]
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
            texto_caixa(frame, status, (15, h - 62), escala=0.75, cor=(80, 180, 255))

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
        texto_caixa(frame, "Q: sair", (15, h - 28), escala=0.6, cor=(220, 220, 220))

        cv2.imshow("S.I.N.A.I.S - Reconhecimento de Libras", frame)

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
    parser.add_argument('--min_maos', type=float, default=0.3,
                        help='Fração mínima de frames com mão no buffer p/ tentar prever. '
                             'Padrão: 0.3 (sinais rápidos têm mão em ~15 dos 30 frames)')
    parser.add_argument('--unificar', default='QUE',
                        help='Palavras cujas variantes numeradas (QUE1/QUE2) são unificadas '
                             'na exibição, somando probabilidades. Separadas por vírgula; '
                             "'' desativa. Padrão: QUE")
    main(parser.parse_args())
