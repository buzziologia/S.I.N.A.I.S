"""
Aquisição de vídeos de teste pela webcam - grava VOCÊ sinalizando, para depois
testar / fazer fine-tune do modelo nas suas próprias amostras (resolve o domain
shift: o modelo passa a ver a SUA execução, não só a do dicionário INES).

Para cada palavra grava N repetições. Os vídeos são salvos com o frame CRU
(não-espelhado), igual à orientação de data/raw_videos - o extract_features.py
espelha na hora de extrair, então a consistência treino/inferência é mantida.
O preview na tela é espelhado só pra ficar natural pra você.

Layout: um PAINEL no topo mostra os comandos, a configuração de mão e o vídeo de
referência (com velocidade ajustável); a câmera fica embaixo, sem sobreposição.
Cada rep mostra uma DICA DE VARIAÇÃO (mais perto/longe/rápido/ângulo) - variar
vale mais que repetir igual. Com --holdout N, as ÚLTIMAS N reps de cada palavra
vão para uma pasta de teste separada (não usar no treino → avaliação honesta).
Após gravar, você REVISA o take e escolhe salvar ou refazer.

Uso:
    python scripts/training/coletar_videos.py --palavras CASA,AMOR,EU --reps 8
    python scripts/training/coletar_videos.py --lista palavras.txt --reps 7 --holdout 2
    python scripts/training/coletar_videos.py --top_k 300 --reps 8 --holdout 2
    python scripts/training/coletar_videos.py --top_k 11,20 --reps 8   # ranks 11 a 20

Controles:
    Espera : ESPAÇO grava | V cicla velocidade da referência (1→0.75→0.5) | P pula | Q sai
    Revisão: S/ENTER salva | R refaz a gravação | Q sai
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
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALTURA_PAINEL = 190   # altura (px) do painel preto acima da câmera
LARG_LISTA    = 190   # largura (px) da coluna lateral com a lista de palavras


def tem_display():
    # DISPLAY/WAYLAND_DISPLAY são conceitos do X11/Wayland e só existem no Linux.
    # Windows e macOS sempre têm display em sessão interativa - sem esta exceção,
    # a checagem desligava o preview da câmera no Windows.
    if sys.platform in ('win32', 'darwin'):
        return True
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def mostrar_frame(janela, frame):
    if not tem_display():
        return
    cv2.imshow(janela, frame)

# Conjunto padrão (palavras comuns e distintas, presentes no top-100/300)
PALAVRAS_PADRAO = ['OI', 'OBRIGADO1', 'POR_FAVOR', 'EU', 'VOCE', 'CASA',
                   'AMOR', 'AGUA', 'AMIGO', 'SIM']

# Dicas de variação por repetição (variação > contagem: cada rep diferente vale mais)
DICAS_VARIACAO = ['normal, centralizado', 'mais perto da camera', 'mais longe',
                  'um pouco mais rapido', 'leve angulo / outra posicao']


def dica_variacao(rep_idx, is_holdout):
    if is_holdout:
        return 'HOLD-OUT (teste) - execucao natural'
    return DICAS_VARIACAO[rep_idx % len(DICAS_VARIACAO)]


def _carregar_top_k(caminho_csv, spec):
    """
    Palavras do CSV de frequência (ranquear_palavras.py) por ranking.
    spec: 'K' → top-K (ranks 1..K)  |  'INI,FIM' → ranks INI..FIM (1-based, inclusivo).
    Ex: '300' → as 300 mais usadas; '11,20' → da 11ª à 20ª (pula as 10 primeiras).
    """
    import csv
    if not os.path.exists(caminho_csv):
        print(f"[ERRO] CSV de frequência não encontrado: {caminho_csv}. "
              "Rode scripts/utils/ranquear_palavras.py primeiro.")
        sys.exit(1)
    partes = [p.strip() for p in str(spec).split(',') if p.strip()]
    try:
        if len(partes) == 1:
            ini, fim = 1, int(partes[0])
        elif len(partes) == 2:
            ini, fim = int(partes[0]), int(partes[1])
        else:
            raise ValueError
        if ini < 1 or fim < ini:
            raise ValueError
    except ValueError:
        print(f"[ERRO] --top_k inválido: '{spec}'. Use 'K' (ex: 300) ou 'INI,FIM' (ex: 11,20).")
        sys.exit(1)
    palavras = []
    with open(caminho_csv, encoding='utf-8') as f:
        for rank, linha in enumerate(csv.DictReader(f), start=1):
            if rank < ini:
                continue
            if rank > fim:
                break
            palavras.append(linha['palavra'])
    print(f"[TOP_K] ranks {ini}..{fim} → {len(palavras)} palavras")
    return palavras


def ler_palavras(args):
    if args.lista:
        with open(args.lista, encoding='utf-8') as f:
            return [l.strip() for l in f if l.strip()]
    if args.palavras:
        return [p.strip() for p in args.palavras.split(',') if p.strip()]
    if args.top_k:
        return _carregar_top_k(args.freq_csv, args.top_k)
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


def achar_config_mao(ref_path):
    """A configuracao_mao.jpg fica na mesma pasta do vídeo de referência."""
    if not ref_path:
        return None
    p = os.path.join(os.path.dirname(ref_path), 'configuracao_mao.jpg')
    return cv2.imread(p) if os.path.exists(p) else None


class RefPlayer:
    """Vídeo de referência em loop, com velocidade ajustável. tick() devolve o frame atual."""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path) if path else None
        self.acc = 0.0
        self.last = None

    def _avancar(self):
        ok, f = self.cap.read()
        if not ok:                                # acabou → reinicia (loop)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        if ok:
            self.last = f

    def tick(self, velocidade=1.0):
        if self.cap is None:
            return None
        if self.last is None:
            self._avancar()
        else:                                     # avança < 1 frame/iteração se lento
            self.acc += velocidade
            while self.acc >= 1.0:
                self.acc -= 1.0
                self._avancar()
        return self.last

    def release(self):
        if self.cap is not None:
            self.cap.release()


def sem_acento(txt: str) -> str:
    """
    Remove acentos p/ exibição no OpenCV ('NÃO1' → 'NAO1'). As fontes Hershey
    do cv2.putText só têm glifos ASCII - acentuados viram '??' na tela.
    Só afeta a RENDERIZAÇÃO; nomes de pastas/arquivos mantêm o acento.
    """
    import unicodedata
    return unicodedata.normalize('NFKD', txt).encode('ascii', 'ignore').decode('ascii')


def _colar(painel, img, x_dir, label, cor, h_alvo):
    """Cola uma miniatura (alinhada à direita em x_dir) com borda+rótulo. Retorna o novo x_dir."""
    ww = int(img.shape[1] * h_alvo / img.shape[0])
    x1, y1 = x_dir - ww, 36
    painel[y1:y1 + h_alvo, x1:x1 + ww] = cv2.resize(img, (ww, h_alvo))
    cv2.rectangle(painel, (x1, y1), (x1 + ww, y1 + h_alvo), cor, 3)
    cv2.putText(painel, label, (x1, y1 - 9), FONT, 0.55, cor, 2, cv2.LINE_AA)
    return x1 - 16


def montar_painel(W, linhas, config_img, ref_frame, velocidade):
    """Painel preto (acima da câmera): texto à esquerda, config de mão + referência à direita."""
    p = np.zeros((ALTURA_PAINEL, W, 3), dtype=np.uint8)
    x_dir = W - 10
    if ref_frame is not None:
        x_dir = _colar(p, ref_frame, x_dir, f"REFERENCIA {velocidade:g}x",
                       (0, 255, 120), ALTURA_PAINEL - 46)
    if config_img is not None:
        x_dir = _colar(p, config_img, x_dir, "CONFIG. MAO",
                       (255, 190, 0), ALTURA_PAINEL - 76)
    y = 46
    for item in linhas:
        txt, cor = item[0], item[1]
        esc = item[2] if len(item) > 2 else 0.7
        cv2.putText(p, sem_acento(txt), (14, y), FONT, esc, cor, 2, cv2.LINE_AA)
        y += 40
    return p


def _icone_check(col, x, y, cor):
    cv2.line(col, (x, y - 3), (x + 4, y + 2), cor, 2, cv2.LINE_AA)
    cv2.line(col, (x + 4, y + 2), (x + 12, y - 8), cor, 2, cv2.LINE_AA)


def _icone_x(col, x, y, cor):
    cv2.line(col, (x, y - 8), (x + 10, y + 2), cor, 2, cv2.LINE_AA)
    cv2.line(col, (x + 10, y - 8), (x, y + 2), cor, 2, cv2.LINE_AA)


def montar_lista(Ws, H, palavras, estado, idx_atual):
    """Coluna lateral com as palavras da sessão: ✓ verde (feito), ✗ laranja (pulado)."""
    col = np.zeros((H, Ws, 3), dtype=np.uint8)
    cv2.putText(col, "PALAVRAS", (12, 26), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    lh, y0 = 24, 52
    n_vis = max(1, (H - y0 - 18) // lh)
    n = len(palavras)
    ini = 0 if n <= n_vis else min(max(idx_atual - n_vis // 2, 0), n - n_vis)
    for k in range(ini, min(ini + n_vis, n)):
        y = y0 + (k - ini) * lh
        st = estado[k]
        nome = sem_acento(palavras[k])
        nome = nome if len(nome) <= 13 else nome[:12] + '.'
        if st == 'feito':
            cor = (0, 255, 120); _icone_check(col, 12, y, cor)
        elif st == 'pulado':
            cor = (0, 140, 255); _icone_x(col, 12, y, cor)
        elif k == idx_atual:
            cor = (0, 255, 255)
            cv2.putText(col, ">", (12, y), FONT, 0.5, cor, 2, cv2.LINE_AA)
        else:
            cor = (170, 170, 170)
        cv2.putText(col, nome, (30, y), FONT, 0.5, cor, 1, cv2.LINE_AA)
    feitos  = sum(1 for s in estado if s == 'feito')
    pulados = sum(1 for s in estado if s == 'pulado')
    cv2.putText(col, f"ok:{feitos}  pul:{pulados}  /{n}", (12, H - 12),
                FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return col


def render(janela, cam, linhas, config_img, ref, velocidade, lista=None):
    """Monta lista lateral + câmera (embaixo) e painel (topo), e mostra."""
    base = cam
    if lista is not None:
        palavras, estado, idx = lista
        base = np.hstack([montar_lista(LARG_LISTA, cam.shape[0], palavras, estado, idx), cam])
    ref_frame = ref.tick(velocidade) if ref is not None else None
    painel = montar_painel(base.shape[1], linhas, config_img, ref_frame, velocidade)
    mostrar_frame(janela, np.vstack([painel, base]))


def main(args):
    palavras = ler_palavras(args)
    n_holdout = min(args.holdout, args.reps)     # últimas N reps de cada palavra → teste
    n_treino  = args.reps - n_holdout
    saida_ho  = args.saida + '_holdout'
    feat_ho   = args.dir_features + '_holdout'
    print(f"Palavras a coletar ({len(palavras)}): {', '.join(palavras)}")
    print(f"Reps/palavra: {args.reps}  (treino={n_treino}, hold-out={n_holdout}) | duração: {args.duracao}s")
    print(f"Saída treino: {args.saida}")
    if n_holdout:
        print(f"Saída teste : {saida_ho}   (hold-out - NÃO usar no treino)")

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

    janela = "S.I.N.A.I.S - Coleta de vídeos"
    if tem_display():
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
        cv2.moveWindow(janela, 0, 0)
    else:
        print("[aviso] sem display detectado; o preview da câmera será desativado.")
    sair = False
    total_gravados = 0
    velocidade = args.ref_velocidade   # velocidade do vídeo de referência (tecla V cicla)
    estado = ['pendente'] * len(palavras)   # status por palavra: 'pendente' | 'feito' | 'pulado'

    for idx_palavra, palavra in enumerate(palavras):
        if sair:
            break
        base = palavra.lower()
        pasta_tr = os.path.join(args.saida, palavra)
        os.makedirs(pasta_tr, exist_ok=True)
        pasta_ho = os.path.join(saida_ho, palavra)
        if n_holdout:
            os.makedirs(pasta_ho, exist_ok=True)
        ja_tr = len(glob.glob(os.path.join(pasta_tr, f"{base}_meu_*.mp4")))
        ja_ho = len(glob.glob(os.path.join(pasta_ho, f"{base}_meu_*.mp4")))
        feitos_tr = feitos_ho = 0

        # Referência do dicionário: vídeo (em loop) + foto da configuração de mão
        ref_path   = achar_video_referencia(palavra)
        ref        = RefPlayer(ref_path)
        config_img = achar_config_mao(ref_path)
        if ref_path is None:
            print(f"  [aviso] sem vídeo de referência p/ '{palavra}' (confira o nome/acento).")
        lista = (palavras, estado, idx_palavra)   # p/ a coluna lateral

        rep = 0
        while rep < args.reps:
            if sair:
                break

            is_holdout = rep >= n_treino
            dica       = dica_variacao(rep, is_holdout)
            cor_dica   = (0, 180, 255) if is_holdout else (0, 255, 255)

            # ── Tela de espera (ESPAÇO grava; V cicla velocidade da referência) ─
            comecar = False
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                cam = cv2.flip(frame, 1)
                linhas = [
                    (f"Palavra: {palavra}   (rep {rep + 1}/{args.reps})", (0, 255, 120), 0.7),
                    (f"Variacao: {dica}", cor_dica, 0.6),
                    (f"ESPACO grava | V vel:{velocidade:g}x | P pula | Q sai", (200, 200, 200), 0.5),
                ]
                render(janela, cam, linhas, config_img, ref, velocidade, lista)
                k = cv2.waitKey(1) & 0xFF
                if k == ord(' '):
                    comecar = True
                    break
                if k in (ord('v'), ord('V')):
                    velocidade = {1.0: 0.75, 0.75: 0.5, 0.5: 1.0}.get(velocidade, 1.0)
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
                    cam = cv2.flip(frame, 1)
                    cv2.putText(cam, str(n), (W // 2 - 30, H // 2),
                                FONT, 4.0, (0, 255, 255), 6, cv2.LINE_AA)
                    linhas = [(f"{palavra}: prepare-se...", (0, 255, 255), 0.7),
                              (f"Variacao: {dica}", cor_dica, 0.6)]
                    render(janela, cam, linhas, config_img, ref, velocidade, lista)
                    cv2.waitKey(1)

            # ── Gravação para a MEMÓRIA (frames crus + features) ───────────────
            frames_raw, frames_feat, abortou = [], [], False
            t0 = time.time()
            while time.time() - t0 < args.duracao:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_raw.append(frame.copy())          # CRU (salvo só se mantiver)
                prog = (time.time() - t0) / args.duracao
                cam = cv2.flip(frame, 1)
                # features do frame espelhado (igual ao extract_features, que espelha)
                res = hands.process(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB))
                frames_feat.append(ef.frame_para_vetor(res))
                cv2.circle(cam, (30, 30), 12, (0, 0, 255), -1)   # REC
                cv2.putText(cam, "REC", (50, 38), FONT, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(cam, (10, H - 25), (10 + int((W - 20) * prog), H - 12),
                              (0, 0, 255), -1)
                linhas = [(f"Gravando: {palavra}", (0, 255, 120), 0.7),
                          (f"Variacao: {dica}", cor_dica, 0.6)]
                render(janela, cam, linhas, config_img, ref, velocidade, lista)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                    abortou = True
                    break
            if abortou:
                sair = True
                break
            if not frames_raw:
                continue                                 # nada gravado → tenta de novo

            # ── Revisão: toca o que foi gravado (S salva / R refaz / Q sai) ────
            decisao, i = None, 0
            while True:
                cam = cv2.flip(frames_raw[i % len(frames_raw)], 1)
                linhas = [(f"{palavra}: revisao (rep {rep + 1})", (0, 255, 120), 0.7),
                          ("S/ENTER salvar | R refazer | Q sair", (200, 200, 200), 0.5)]
                render(janela, cam, linhas, config_img, ref, velocidade, lista)
                i += 1
                k = cv2.waitKey(40) & 0xFF               # ~25 fps no playback
                if k in (ord('s'), ord('S'), 13):
                    decisao = 'salvar'
                    break
                if k in (ord('r'), ord('R')):
                    decisao = 'refazer'
                    break
                if k in (ord('q'), ord('Q'), 27):
                    decisao = 'sair'
                    break
            if decisao == 'sair':
                sair = True
                break
            if decisao == 'refazer':
                print(f"  ↻ rep {rep + 1} refeita")
                continue                                 # regrava a MESMA rep

            # ── Salva (vídeo CRU + features 126-d) ─────────────────────────────
            if is_holdout:
                pasta_dest, feat_root, idx = pasta_ho, feat_ho, ja_ho + feitos_ho
            else:
                pasta_dest, feat_root, idx = pasta_tr, args.dir_features, ja_tr + feitos_tr
            caminho = os.path.join(pasta_dest, f"{base}_meu_{idx:02d}.mp4")
            writer = cv2.VideoWriter(caminho, fourcc, args.fps, (W, H))
            for fr in frames_raw:
                writer.write(fr)
            writer.release()
            if frames_feat:
                pasta_f = os.path.join(feat_root, palavra)
                os.makedirs(pasta_f, exist_ok=True)
                np.save(os.path.join(pasta_f, f"{base}_meu_{idx:02d}.npy"),
                        np.array(frames_feat, dtype=np.float32))
            if is_holdout:
                feitos_ho += 1
            else:
                feitos_tr += 1
            total_gravados += 1
            rep += 1
            tag = " [HOLD-OUT]" if is_holdout else ""
            print(f"  ✓ {caminho}{tag}  ({len(frames_feat)} frames → .npy)")

        if not sair:
            estado[idx_palavra] = 'feito' if rep >= args.reps else 'pulado'
        ref.release()

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print(f"\n[OK] {total_gravados} take(s) salvos:")
    print(f"  treino : {args.saida}/PALAVRA/*.mp4  (+ {args.dir_features}/PALAVRA/*.npy)")
    if n_holdout:
        print(f"  teste  : {saida_ho}/PALAVRA/*.mp4  (+ {feat_ho}/...)  ← hold-out, NÃO treinar")
    print("Para TESTAR o modelo na sua sinalização: avaliar os .npy num modelo treinado.")
    print("Para FINE-TUNE: copie os .mp4 de TREINO p/ data/raw_videos/<ASSUNTO>/ e rode extract_features → train.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coleta de vídeos de Libras pela webcam')
    parser.add_argument('--palavras', default=None,
                        help='Lista separada por vírgula (ex: CASA,AMOR,EU)')
    parser.add_argument('--lista', default=None,
                        help='Arquivo .txt com uma palavra por linha')
    parser.add_argument('--top_k', default=None,
                        help="Coletar por ranking de uso: 'K' (ex: 300 = as 300 mais usadas) "
                             "ou 'INI,FIM' (ex: 11,20 = da 11a a 20a mais usada)")
    parser.add_argument('--freq_csv', default='data/palavras_por_frequencia.csv',
                        help='CSV de frequência gerado por ranquear_palavras.py')
    parser.add_argument('--reps', type=int, default=5, help='Repetições por palavra')
    parser.add_argument('--holdout', type=int, default=0,
                        help='Das reps, quantas (as últimas) vão pro conjunto de teste (hold-out)')
    parser.add_argument('--duracao', type=float, default=2.5, help='Segundos por gravação')
    parser.add_argument('--ref_velocidade', type=float, default=1.0,
                        help='Velocidade inicial do vídeo de referência (0.5, 0.75, 1.0). Tecla V cicla.')
    parser.add_argument('--saida', default='data/meus_videos', help='Pasta de saída dos .mp4')
    parser.add_argument('--dir_features', default='data/meus_features',
                        help='Pasta de saída das features .npy 126-dim')
    parser.add_argument('--camera', type=int, default=0, help='Índice da câmera')
    parser.add_argument('--fps', type=int, default=20, help='FPS gravado no arquivo')
    main(parser.parse_args())
