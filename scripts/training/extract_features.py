import cv2
import mediapipe as mp
import os
import glob
import numpy as np # Nova biblioteca para salvar em .npy
from datetime import datetime

# Configurações do MediaPipe para VÍDEOS
mp_hands = mp.solutions.hands


def criar_hands():
    """
    Cria um rastreador de mãos novo. Um por vídeo: static_image_mode=False
    mantém estado de rastreamento entre frames, e reusar o mesmo objeto entre
    vídeos diferentes 'vaza' o rastreio do vídeo anterior para o seguinte.
    max_num_hands=2 → captura as duas mãos (muitos sinais de Libras são bimanuais).
    """
    return mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                          min_detection_confidence=0.5)

# Cada frame vira um vetor de 126 = 2 mãos × 21 landmarks × XYZ.
#   Slot 0 (índices  0:63 ) = mão rotulada "Left"  pelo MediaPipe
#   Slot 1 (índices 63:126) = mão rotulada "Right"
# Mão ausente no frame → bloco de 63 zeros (o modelo aprende a ignorar).
DIM_MAO   = 63
NUM_MAOS  = 2
DIM_FRAME = DIM_MAO * NUM_MAOS  # 126


def frame_para_vetor(results) -> np.ndarray:
    """
    Converte o resultado do MediaPipe num vetor (126,): as duas mãos lado a lado.
    A mão 'Left' ocupa o bloco 0:63 e a 'Right' o bloco 63:126; mão ausente = zeros.
    """
    vetor = np.zeros(DIM_FRAME, dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for mao_lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            rotulo = handed.classification[0].label   # 'Left' ou 'Right'
            base   = (0 if rotulo == 'Left' else 1) * DIM_MAO
            pts = []
            for lm in mao_lm.landmark:
                pts.extend([lm.x, lm.y, lm.z])
            vetor[base:base + DIM_MAO] = np.array(pts, dtype=np.float32)
    return vetor

# Caminhos
DIRS_VIDEOS = ['data/raw_videos']
DIR_NPY = 'data/processed_features'

# Cria a pasta de saída se ela não existir
os.makedirs(DIR_NPY, exist_ok=True)

def extrair_videos_para_npy():
    # Pega todos os vídeos .mp4 dentro da(s) pasta(s) e subpastas
    caminhos_videos = []
    for dir_videos in DIRS_VIDEOS:
        if os.path.isdir(dir_videos):
            caminhos_videos.extend(
                glob.glob(os.path.join(dir_videos, "**", "*.mp4"), recursive=True)
            )

    if not caminhos_videos:
        print(f"Nenhum vídeo encontrado nas pastas {DIRS_VIDEOS}.")
        return

    print(f">>> Iniciando extração de {len(caminhos_videos)} vídeos...")

    tempo_processo = []

    num_videos = len(caminhos_videos)

    for index, caminho_video in enumerate(caminhos_videos, start = 1):
        start_time = datetime.now()

        nome_video = os.path.basename(caminho_video)
        nome_sem_extensao = os.path.splitext(nome_video)[0]
        
        # Onde o arquivo .npy será salvo
        caminho_salvar_npy = os.path.join(DIR_NPY, f"{nome_sem_extensao}.npy")
        
        # Pula se já tiver sido processado antes (economiza tempo)
        if os.path.exists(caminho_salvar_npy):
            continue

        cap = cv2.VideoCapture(caminho_video)
        hands = criar_hands()   # rastreador novo por vídeo (sem estado do anterior)
        frames_do_sinal = [] # Lista que vai guardar a matriz temporal do vídeo

        while cap.isOpened():
            sucesso, frame = cap.read()
            if not sucesso:
                break # O vídeo acabou

            # Espelha o frame (igual ao testar_camera.py) para que os rótulos
            # "Left"/"Right" do MediaPipe sejam consistentes entre o treino e a
            # inferência na webcam, que também é espelhada.
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # 1 vetor (126,) por frame: as duas mãos lado a lado
            frames_do_sinal.append(frame_para_vetor(results))

        cap.release()
        hands.close()
        
        # Converte tudo para uma Matriz NumPy e salva
        matriz_final = np.array(frames_do_sinal)
        np.save(caminho_salvar_npy, matriz_final)
        ellapsed = datetime.now() - start_time

        tempo_processo.append(ellapsed.total_seconds())

        tempo_processo_estimado = sum(tempo_processo) / len(tempo_processo)
        tempo_restante_segundos = (num_videos - index) * tempo_processo_estimado

        tempo_restante = tempo_restante_segundos if tempo_restante_segundos < 60 else tempo_restante_segundos / 60

        print(f"Processando [{index}/{num_videos}]: {nome_video} ({tempo_restante:.0f}{'s' if tempo_restante_segundos < 60 else ' min'} restantes)")

    print("\n[SUCESSO] Todos os vídeos foram convertidos em matrizes .npy!")

if __name__ == "__main__":
    extrair_videos_para_npy()
