import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import base64
from typing import Any

# ==========================================================
# 🎨 1. Estilização Premium (Glassmorphism & Neon Dark Mode)
# ==========================================================
def apply_premium_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        /* Fontes e fundo global */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            background-color: #0E1117;
            color: #E0E2E6;
        }
        
        /* Estilo da barra lateral */
        [data-testid="stSidebar"] {
            background-color: #161A22;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Cartão Glassmorphic para Exibição da Tradução */
        .translation-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            padding: 24px;
            text-align: center;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
            margin: 20px 0px;
            transition: all 0.3s ease;
        }
        .translation-card:hover {
            border: 1px solid rgba(0, 242, 254, 0.3);
            box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.1);
        }
        
        .translation-title {
            font-size: 0.95rem;
            color: #8892B0;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 8px;
            font-weight: 600;
        }
        
        .translation-word {
            font-size: 2.8rem;
            font-weight: 700;
            color: #00F2FE;
            text-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
        }
        
        .fps-badge {
            display: inline-block;
            background: rgba(0, 242, 254, 0.1);
            color: #00F2FE;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ==========================================================
# 🔊 2. Sintetizador de Áudio Não-Bloqueante (gTTS + HTML)
# ==========================================================
def play_audio_non_blocking(text: str):
    try:
        from gtts import gTTS
        import io
        tts = gTTS(text=text, lang="pt")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64_audio = base64.b64encode(fp.read()).decode("utf-8")
        audio_tag = f'<audio autoplay src="data:audio/mp3;base64,{b64_audio}">'
        st.markdown(audio_tag, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Erro na síntese de áudio: {e}")

# ==========================================================
# 🧮 3. Funções Utilitárias de Rastreamento (Numpy Normalization)
# ==========================================================
def extract_and_normalize_landmarks(landmarks_list: Any) -> np.ndarray:
    """
    Extrai coordenadas x, y, z e realiza translação e escala vetorizada.
    Retorna array achatado de tamanho 63.
    """
    if not landmarks_list:
        return np.zeros(63, dtype=np.float32)
    
    # Extração das coordenadas brutas (21 pontos, 3 dimensões)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list.landmark], dtype=np.float32)
    
    # Translação: Pulso (Landmark 0) vira origem (0, 0, 0)
    anchor = coords[0]
    coords_norm = coords - anchor
    
    # Escalonamento: Divide pela distância até a ponta do dedo médio (Landmark 12)
    dist_max = np.linalg.norm(coords_norm[12])
    if dist_max > 1e-6:
        coords_scaled = coords_norm / dist_max
    else:
        coords_scaled = coords_norm
        
    return coords_scaled.flatten()

# ==========================================================
# 🖥️ 4. Layout e Interface do Usuário (Streamlit App)
# ==========================================================
def main():
    st.set_page_config(page_title="S.I.N.A.I.S - Tradutor de LIBRAS", layout="wide", page_icon="🤟")
    apply_premium_style()
    
    # --- Sidebar de Informações ---
    st.sidebar.markdown("<h2 style='text-align: center; color: #00F2FE;'>🤟 S.I.N.A.I.S</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center; color: #8892B0; font-size:0.9rem;'>Sistema Integrado de Tradução e Processamento de Sinais</p>", unsafe_allow_html=True)
    st.sidebar.write("---")
    st.sidebar.markdown("### 👥 Equipe de Desenvolvimento")
    st.sidebar.markdown(
        """
        - 👤 **Ana Clara Guimarães**
        - 👤 **Mateus Bueno Ferreira**
        - 👤 **Sofia Schmitz**
        - 👤 **Vinícios Buzzi**
        """
    )
    st.sidebar.write("---")
    st.sidebar.markdown("### 📚 Padrão Ouro")
    st.sidebar.info("Dicionário Oficial de LIBRAS do **INES** (Instituto Nacional de Educação de Surdos).")

    # --- Título Principal ---
    st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 0px;'>Tradutor de LIBRAS em Tempo Real</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8892B0; font-size: 1.1rem; margin-bottom: 30px;'>Pipeline Inteligente de Captura, Extração de Landmarks e Inferência Temporal</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("🎥 Feed da Webcam")
        # Controles da Câmera
        run_camera = st.checkbox("Ativar Câmera", value=True)
        FRAME_WINDOW = st.image([]) # Elemento de imagem do Streamlit para o stream de vídeo
        
    with col2:
        st.subheader("🤟 Tradução & Métricas")
        
        # Cartão Glassmorphism de Tradução
        st.markdown(
            """
            <div class="translation-card">
                <div class="translation-title">Sinal Detectado</div>
                <div class="translation-word" id="word-display">AGUARDANDO...</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Métricas Dinâmicas
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("FPS Estimado", "30 Hz")
        with m_col2:
            st.metric("Confiança de Inferência", "0.0%")
            
        st.write("---")
        st.subheader("🔊 Voz Sintetizada (gTTS)")
        speak_btn = st.button("🔊 Ouvir Tradução", use_container_width=True)
        if speak_btn:
            play_audio_non_blocking("Seja bem-vindo ao S.I.N.A.I.S")

    # --- 5. Rastreamento e Loop da Câmera ---
    if run_camera:
        # Inicializa o MediaPipe Hands
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Câmera não pôde ser aberta. Verifique se ela está conectada ou em uso por outro aplicativo.")
        
        while cap.isOpened() and run_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("Falha na captura do frame.")
                break
                
            # OpenCV captura em BGR, Streamlit/MediaPipe precisa de RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # Desenha as conexões se houver mãos detectadas
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame_rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(0, 242, 254), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(57, 255, 20), thickness=2, circle_radius=2)
                    )
                    
                    # Demonstração da extração e normalização de características de forma contínua
                    features = extract_and_normalize_landmarks(hand_landmarks)
                    # (Essas features podem ser empilhadas em um buffer temporal para alimentar a LSTM)
                    
            # Atualiza o elemento de imagem do Streamlit
            FRAME_WINDOW.image(frame_rgb, channels="RGB")
            
        cap.release()

if __name__ == "__main__":
    main()
