import cv2
import mediapipe as mp
import os
import glob
import numpy as np # Nova biblioteca para salvar em .npy

# Configurações do MediaPipe para VÍDEOS
mp_hands = mp.solutions.hands
# static_image_mode=False diz ao MediaPipe para rastrear o movimento
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Caminhos
DIR_VIDEOS = 'data/raw_videos'
DIR_NPY = 'data/processed_features'

# Cria a pasta de saída se ela não existir
os.makedirs(DIR_NPY, exist_ok=True)

def extrair_videos_para_npy():
    # Pega todos os vídeos .mp4 dentro da pasta e subpastas
    caminhos_videos = glob.glob(os.path.join(DIR_VIDEOS, "**", "*.mp4"), recursive=True)
    
    if not caminhos_videos:
        print(f"Nenhum vídeo encontrado na pasta {DIR_VIDEOS}.")
        return

    print(f">>> Iniciando extração de {len(caminhos_videos)} vídeos...")

    for caminho_video in caminhos_videos:
        nome_video = os.path.basename(caminho_video)
        nome_sem_extensao = os.path.splitext(nome_video)[0]
        
        # Onde o arquivo .npy será salvo
        caminho_salvar_npy = os.path.join(DIR_NPY, f"{nome_sem_extensao}.npy")
        
        # Pula se já tiver sido processado antes (economiza tempo)
        if os.path.exists(caminho_salvar_npy):
            continue
            
        print(f"Processando: {nome_video}...")
        
        cap = cv2.VideoCapture(caminho_video)
        frames_do_sinal = [] # Lista que vai guardar a matriz temporal do vídeo
        
        while cap.isOpened():
            sucesso, frame = cap.read()
            if not sucesso:
                break # O vídeo acabou
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # Vetor de 63 posições para este frame específico
            vetor_pontos = np.zeros(63) 
            
            if results.multi_hand_landmarks:
                # Pega a primeira mão detectada
                hand_landmarks = results.multi_hand_landmarks[0]
                temp_pontos = []
                for lm in hand_landmarks.landmark:
                    temp_pontos.extend([lm.x, lm.y, lm.z])
                vetor_pontos = np.array(temp_pontos)
            
            # Adiciona os pontos deste frame na lista do vídeo
            frames_do_sinal.append(vetor_pontos)
            
        cap.release()
        
        # Converte tudo para uma Matriz NumPy e salva
        matriz_final = np.array(frames_do_sinal)
        np.save(caminho_salvar_npy, matriz_final)

    print("\n[SUCESSO] Todos os vídeos foram convertidos em matrizes .npy!")

if __name__ == "__main__":
    extrair_videos_para_npy()
