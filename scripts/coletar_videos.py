"""
Aquisição de vídeos de teste pela webcam — grava VOCÊ sinalizando, para depois
testar / fazer fine-tune do modelo nas suas próprias amostras (resolve o domain
shift: o modelo passa a ver a SUA execução, não só a do dicionário INES).

Para cada palavra grava N repetições. Os vídeos são salvos com o frame CRU
(não-espelhado), igual à orientação de data/raw_videos — o extract_features.py
espelha na hora de extrair, então a consistência treino/inferência é mantida.
O preview na tela é espelhado só pra ficar natural pra você.

Uso:
    python scripts/coletar_videos.py --palavras CASA,AMOR,EU --reps 5
    python scripts/coletar_videos.py --lista palavras.txt --duracao 3
    python scripts/coletar_videos.py            # usa um conjunto padrão de teste

Controles (durante a coleta):
    ESPAÇO = gravar a repetição atual
    P      = pular a palavra atual
    Q/ESC  = sair
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import extract_features as ef   # reusa frame_para_vetor → 126-dim idêntico ao treino
import mediapipe as mp

mp_hands = mp.solutions.hands

# Conjunto padrão (palavras comuns e distintas, presentes no top-100/300)
PALAVRAS_PADRAO = ['OI', 'OBRIGADO1', 'POR_FAVOR', 'EU', 'VOCE', 'CASA',
                   'AMOR', 'AGUA', 'AMIGO', 'SIM']


def barra_texto(frame, linhas, y0=40, cor=(0, 255, 120)):
    """Desenha um bloco de texto com fundo semitransparente no topo."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30 + 34 * len(linhas)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (txt, c) in enumerate(linhas):
        cv2.putText(frame, txt, (15, y0 + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)


def ler_palavras(args):
    if args.lista:
        with open(args.lista, encoding='utf-8') as f:
            return [l.strip() for l in f if l.strip()]
    if args.palavras:
        return [p.strip() for p in args.palavras.split(',') if p.strip()]
    return PALAVRAS_PADRAO


def achar_video_referencia(palavra, dir_videos='data/raw_videos'):
    """Acha o 1º .mp4 de raw_videos/<ASSUNTO>/<PALAVRA>/ (None se não houver)."""
    if not os.path.isdir(dir_videos):
        return None
    for assunto in os.listdir(dir_videos):
        pp = os.path.join(dir_videos, assunto, palavra)
        if os.path.isdir(pp):
            mp4s = [f for f in os.listdir(pp) if f.lower().endswith('.mp4')]
            if mp4s:
                return os.path.join(pp, mp4s[0])
    return None


def overlay_referencia(preview, ref_cap):
    """Toca o vídeo de referência do dicionário em loop, no canto inferior direito."""
    if ref_cap is None:
        return
    ok, f = ref_cap.read()
    if not ok:                                   # acabou → reinicia (loop contínuo)
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, f = ref_cap.read()
    if not ok:
        return
    H, W  = preview.shape[:2]
    tw    = W // 4
    th    = min(int(f.shape[0] * tw / f.shape[1]), H - 50)
    thumb = cv2.resize(f, (tw, th))
    x1, y1 = W - tw - 10, H - th - 35
    preview[y1:y1 + th, x1:x1 + tw] = thumb
    cv2.rectangle(preview, (x1, y1), (x1 + tw, y1 + th), (0, 255, 120), 2)
    cv2.putText(preview, "referencia", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1, cv2.LINE_AA)


def main(args):
    palavras = ler_palavras(args)
    print(f"Palavras a coletar ({len(palavras)}): {', '.join(palavras)}")
    print(f"Repetições por palavra: {args.reps} | duração: {args.duracao}s | saída: {args.saida}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERRO] Não foi possível abrir a câmera.")
        sys.exit(1)
    ok, frame = cap.read()
    if not ok:
        print("[ERRO] Não foi possível ler da câmera.")
        sys.exit(1)
    H, W = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # MediaPipe p/ extrair as features 126-dim na hora (mesma config do extract_features)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    os.makedirs(args.dir_features, exist_ok=True)

    janela = "S.I.N.A.I.S — Coleta de vídeos"
    sair = False
    total_gravados = 0

    for palavra in palavras:
        if sair:
            break
        pasta = os.path.join(args.saida, palavra)
        os.makedirs(pasta, exist_ok=True)
        base = palavra.lower()
        ja_existem = len(glob.glob(os.path.join(pasta, f"{base}_meu_*.mp4")))

        # Vídeo de referência do dicionário (toca em loop enquanto coleta a palavra)
        ref_path = achar_video_referencia(palavra)
        ref_cap  = cv2.VideoCapture(ref_path) if ref_path else None
        if ref_path is None:
            print(f"  [aviso] sem vídeo de referência p/ '{palavra}' (confira o nome/acento).")

        rep = 0
        while rep < args.reps:
            if sair:
                break

            # ── Tela de espera (aguarda ESPAÇO) ────────────────────────────────
            comecar = False
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                preview = cv2.flip(frame, 1)
                barra_texto(preview, [
                    (f"Palavra: {palavra}   (rep {rep + 1}/{args.reps})", (0, 255, 120)),
                    ("ESPACO: gravar   |   P: pular palavra   |   Q: sair", (200, 200, 200)),
                ])
                overlay_referencia(preview, ref_cap)
                cv2.imshow(janela, preview)
                k = cv2.waitKey(1) & 0xFF
                if k == ord(' '):
                    comecar = True
                    break
                if k in (ord('p'), ord('P')):
                    break               # pula a palavra
                if k in (ord('q'), ord('Q'), 27):
                    sair = True
                    break
            if not comecar:
                break                   # pulou a palavra ou saiu

            # ── Contagem regressiva ────────────────────────────────────────────
            for n in (3, 2, 1):
                t0 = time.time()
                while time.time() - t0 < 0.7:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    preview = cv2.flip(frame, 1)
                    cv2.putText(preview, str(n), (W // 2 - 30, H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 255), 6, cv2.LINE_AA)
                    barra_texto(preview, [(f"{palavra}: prepare-se...", (0, 255, 255))])
                    overlay_referencia(preview, ref_cap)
                    cv2.imshow(janela, preview)
                    cv2.waitKey(1)

            # ── Gravação (salva frame CRU; preview espelhado) ──────────────────
            idx = ja_existem + rep
            caminho = os.path.join(pasta, f"{base}_meu_{idx:02d}.mp4")
            writer = cv2.VideoWriter(caminho, fourcc, args.fps, (W, H))
            frames_feat = []                         # features 126-dim deste take
            t0 = time.time()
            while time.time() - t0 < args.duracao:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)                 # CRU, não-espelhado
                prog = (time.time() - t0) / args.duracao
                preview = cv2.flip(frame, 1)
                # features do frame espelhado (igual ao extract_features, que espelha)
                res = hands.process(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
                frames_feat.append(ef.frame_para_vetor(res))
                cv2.circle(preview, (30, 30), 12, (0, 0, 255), -1)   # REC
                cv2.putText(preview, "REC", (50, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(preview, (10, H - 25), (10 + int((W - 20) * prog), H - 12),
                              (0, 0, 255), -1)
                cv2.putText(preview, palavra, (W // 2 - 60, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2, cv2.LINE_AA)
                overlay_referencia(preview, ref_cap)
                cv2.imshow(janela, preview)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                    sair = True
                    break
            writer.release()
            # Salva as features 126-dim (prontas p/ testar/fine-tune sem re-extrair)
            if frames_feat:
                pasta_f = os.path.join(args.dir_features, palavra)
                os.makedirs(pasta_f, exist_ok=True)
                np.save(os.path.join(pasta_f, f"{base}_meu_{idx:02d}.npy"),
                        np.array(frames_feat, dtype=np.float32))
            total_gravados += 1
            rep += 1
            print(f"  ✓ {caminho}  ({len(frames_feat)} frames → .npy)")

        if ref_cap is not None:
            ref_cap.release()

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print(f"\n[OK] {total_gravados} take(s) salvos:")
    print(f"  vídeos   : {args.saida}/PALAVRA/*.mp4   (cru, formato raw_videos)")
    print(f"  features : {args.dir_features}/PALAVRA/*.npy   (126-dim, prontas p/ teste)")
    print("Para TESTAR o modelo na sua sinalização: avaliar os .npy num modelo treinado.")
    print("Para FINE-TUNE: copie os .mp4 p/ data/raw_videos/<ASSUNTO>/ e rode extract_features → train.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coleta de vídeos de Libras pela webcam')
    parser.add_argument('--palavras', default=None,
                        help='Lista separada por vírgula (ex: CASA,AMOR,EU)')
    parser.add_argument('--lista', default=None,
                        help='Arquivo .txt com uma palavra por linha')
    parser.add_argument('--reps', type=int, default=5, help='Repetições por palavra')
    parser.add_argument('--duracao', type=float, default=2.5, help='Segundos por gravação')
    parser.add_argument('--saida', default='data/meus_videos', help='Pasta de saída dos .mp4')
    parser.add_argument('--dir_features', default='data/meus_features',
                        help='Pasta de saída das features .npy 126-dim')
    parser.add_argument('--camera', type=int, default=0, help='Índice da câmera')
    parser.add_argument('--fps', type=int, default=20, help='FPS gravado no arquivo')
    main(parser.parse_args())
