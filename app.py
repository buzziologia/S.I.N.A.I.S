import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import base64
from typing import Any

# ══════════════════════════════════════════════════════════════════════════════
# 1. Design System — Cyber-Premium Glassmorphism
# ══════════════════════════════════════════════════════════════════════════════

def apply_cyber_premium_style() -> None:
    st.markdown("""
    <style>
    /* ── Fonts ─────────────────────────────────────────────────────────────── */
    @import url('https://api.fontshare.com/v2/css?f[]=cabinet-grotesk@800,700&f[]=satoshi@400,500,700&display=swap');

    /* ── Base ───────────────────────────────────────────────────────────────── */
    html, body, [class*="css"], .stApp {
        background-color: #0e0e0e !important;
        color: #e0e2e6 !important;
        font-family: 'Satoshi', sans-serif !important;
    }

    /* ── Floating atmosphere orbs ──────────────────────────────────────────── */
    .orb-purple {
        position: fixed; top: -150px; left: -150px;
        width: 600px; height: 600px;
        background: #6a48f2; border-radius: 50%;
        filter: blur(100px); opacity: 0.18; z-index: 0;
        animation: float 15s ease-in-out infinite;
        pointer-events: none;
    }
    .orb-pink {
        position: fixed; bottom: -150px; right: -150px;
        width: 500px; height: 500px;
        background: #d946ef; border-radius: 50%;
        filter: blur(100px); opacity: 0.18; z-index: 0;
        animation: float 15s ease-in-out infinite reverse;
        pointer-events: none;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px) scale(1); }
        50%       { transform: translateY(-40px) scale(1.1); }
    }

    /* ── Sidebar ────────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.03) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
        padding-top: 20px;
    }

    /* ── Header ─────────────────────────────────────────────────────────────── */
    header[data-testid="stHeader"] {
        background: rgba(14,14,14,0.85) !important;
        backdrop-filter: blur(20px) !important;
        border-bottom: 1px solid rgba(255,255,255,0.05) !important;
        height: 64px !important;
    }

    /* ── Main container ─────────────────────────────────────────────────────── */
    .main .block-container {
        background: transparent !important;
        padding-top: 1.5rem;
        position: relative; z-index: 1;
    }

    /* ── Typography ─────────────────────────────────────────────────────────── */
    h1, h2, h3 {
        font-family: 'Cabinet Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-style: italic !important;
    }

    /* ── Custom scrollbar ───────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    /* ── Glassmorphic card ──────────────────────────────────────────────────── */
    .glass-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 28px;
    }

    /* ── Lime translation bubble ────────────────────────────────────────────── */
    .translation-bubble {
        background: #bef264;
        color: #1a2e05;
        border-radius: 20px 4px 20px 20px;
        padding: 28px 36px;
        font-family: 'Cabinet Grotesk', sans-serif;
        font-weight: 800;
        font-style: italic;
        font-size: 2.6rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(190,242,100,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin: 12px 0;
        line-height: 1.1;
    }
    .translation-bubble:hover {
        transform: scale(1.02);
        box-shadow: 0 16px 40px rgba(190,242,100,0.3);
    }
    .translation-label {
        font-family: 'Satoshi', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: rgba(255,255,255,0.35);
        margin-bottom: 6px;
    }

    /* ── Gradient action button ─────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6a48f2 0%, #d946ef 100%) !important;
        color: white !important;
        font-family: 'Cabinet Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-style: italic !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 14px 32px !important;
        box-shadow: 0 20px 40px rgba(106,72,242,0.3) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 28px 50px rgba(106,72,242,0.45) !important;
    }

    /* ── Metrics ────────────────────────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        backdrop-filter: blur(20px) !important;
    }
    [data-testid="metric-container"] label {
        color: rgba(255,255,255,0.4) !important;
        font-family: 'Satoshi', sans-serif !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }
    [data-testid="stMetricValue"] {
        color: #bef264 !important;
        font-family: 'Cabinet Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-style: italic !important;
    }

    /* ── Checkbox ───────────────────────────────────────────────────────────── */
    .stCheckbox label {
        font-family: 'Satoshi', sans-serif !important;
        color: rgba(255,255,255,0.7) !important;
        font-weight: 500 !important;
    }

    /* ── Camera image border ────────────────────────────────────────────────── */
    [data-testid="stImage"] img {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Divider ────────────────────────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.05) !important; }

    /* ── Sidebar custom components ──────────────────────────────────────────── */
    .sidebar-label {
        font-family: 'Satoshi', sans-serif;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: rgba(255,255,255,0.28);
        margin: 20px 0 10px 2px;
        display: block;
    }
    .member-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 10px;
        border-radius: 12px;
        transition: background 0.2s ease;
        margin-bottom: 4px;
        cursor: default;
    }
    .member-item:hover { background: rgba(255,255,255,0.04); }
    .member-avatar {
        width: 40px; height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6a48f2, #d946ef);
        display: flex; align-items: center; justify-content: center;
        font-family: 'Cabinet Grotesk', sans-serif;
        font-weight: 800; font-size: 0.85rem; color: white;
        position: relative; flex-shrink: 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .member-dot {
        position: absolute; bottom: 1px; right: 1px;
        width: 10px; height: 10px;
        background: #10b981; border-radius: 50%;
        border: 2px solid #0e0e0e;
    }
    .member-name {
        font-family: 'Satoshi', sans-serif;
        font-size: 0.88rem; font-weight: 500;
        color: rgba(255,255,255,0.82);
    }
    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.25);
        color: #10b981;
        font-family: 'Satoshi', sans-serif;
        font-size: 0.68rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2px;
        padding: 3px 10px; border-radius: 100px;
    }
    .status-dot-live {
        width: 6px; height: 6px;
        background: #10b981; border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
        display: inline-block;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; } 50% { opacity: 0.3; }
    }
    .gradient-line {
        height: 2px;
        background: linear-gradient(135deg, #6a48f2 0%, #d946ef 100%);
        border-radius: 1px; margin: 16px 0;
    }
    .brand-title {
        font-family: 'Cabinet Grotesk', sans-serif;
        font-weight: 800; font-style: italic;
        font-size: 1.6rem;
        background: linear-gradient(135deg, #6a48f2, #d946ef);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
        margin-bottom: 2px;
    }
    .brand-sub {
        font-family: 'Satoshi', sans-serif;
        font-size: 0.65rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 3px;
        color: rgba(255,255,255,0.3);
    }
    .section-title {
        font-family: 'Cabinet Grotesk', sans-serif;
        font-weight: 800; font-style: italic;
        font-size: 1.1rem; color: white;
        margin-bottom: 12px;
    }
    </style>
    """, unsafe_allow_html=True)


def render_orbs() -> None:
    st.markdown('<div class="orb-purple"></div><div class="orb-pink"></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Síntese de Áudio (gTTS não-bloqueante)
# ══════════════════════════════════════════════════════════════════════════════

def play_audio_non_blocking(text: str) -> None:
    try:
        from gtts import gTTS
        import io
        tts = gTTS(text=text, lang="pt")
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        b64 = base64.b64encode(fp.read()).decode("utf-8")
        st.markdown(f'<audio autoplay src="data:audio/mp3;base64,{b64}">',
                    unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Erro na síntese de áudio: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Extração e Normalização de Landmarks
# ══════════════════════════════════════════════════════════════════════════════

def extract_and_normalize_landmarks(landmarks_list: Any) -> np.ndarray:
    """
    Extrai e normaliza 21 landmarks 3D de uma mão detectada.
    Retorna vetor achatado de tamanho 63 (zeros se mão ausente).
    """
    if not landmarks_list:
        return np.zeros(63, dtype=np.float32)

    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks_list.landmark],
        dtype=np.float32
    )
    # Translação: pulso (landmark 0) como origem
    coords_norm = coords - coords[0]
    # Escalonamento: distância ao dedo médio (landmark 12)
    dist_max = np.linalg.norm(coords_norm[12])
    if dist_max > 1e-6:
        coords_norm = coords_norm / dist_max

    return coords_norm.flatten()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Sidebar
# ══════════════════════════════════════════════════════════════════════════════

TEAM = [
    ("AC", "Ana Clara Guimarães"),
    ("MB", "Mateus Bueno Ferreira"),
    ("SS", "Sofia Schmitz"),
    ("VB", "Vinícios Buzzi"),
]

def render_sidebar() -> None:
    with st.sidebar:
        # Brand
        st.markdown("""
            <div class="brand-title">S.I.N.A.I.S</div>
            <div class="brand-sub">Tradutor de LIBRAS</div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)

        # Status
        st.markdown("""
            <span class="status-badge">
                <span class="status-dot-live"></span>
                Sistema Online
            </span>
        """, unsafe_allow_html=True)

        # Equipe
        st.markdown('<span class="sidebar-label">Equipe de Desenvolvimento</span>',
                    unsafe_allow_html=True)
        for initials, name in TEAM:
            st.markdown(f"""
                <div class="member-item">
                    <div class="member-avatar">
                        {initials}
                        <div class="member-dot"></div>
                    </div>
                    <span class="member-name">{name}</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)

        # Referência
        st.markdown('<span class="sidebar-label">Padrão de Referência</span>',
                    unsafe_allow_html=True)
        st.markdown("""
            <div style="
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 14px;
                padding: 14px 16px;
                font-family: 'Satoshi', sans-serif;
                font-size: 0.82rem;
                color: rgba(255,255,255,0.55);
                line-height: 1.5;
            ">
                📚 Dicionário Oficial de LIBRAS<br>
                <span style="color: rgba(255,255,255,0.25); font-size: 0.72rem;">
                    Instituto Nacional de Educação de Surdos
                </span>
            </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. App Principal
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="S.I.N.A.I.S — Tradutor de LIBRAS",
        layout="wide",
        page_icon="🤟",
    )
    apply_cyber_premium_style()
    render_orbs()
    render_sidebar()

    # ── Título ──────────────────────────────────────────────────────────────
    st.markdown("""
        <h1 style="text-align:center; margin-bottom:4px;">
            S.I.N.A.I.S
        </h1>
        <p style="
            text-align:center;
            font-family:'Satoshi',sans-serif;
            color:rgba(255,255,255,0.38);
            font-size:0.95rem;
            letter-spacing:1px;
            margin-bottom:28px;
        ">
            Captura · Rastreamento · Inferência Temporal
        </p>
    """, unsafe_allow_html=True)

    col_cam, col_info = st.columns([2, 1], gap="large")

    # ── Coluna esquerda: feed da câmera ─────────────────────────────────────
    with col_cam:
        st.markdown('<div class="section-title">🎥 Feed da Webcam</div>',
                    unsafe_allow_html=True)
        run_camera = st.checkbox("Ativar Câmera", value=True)
        FRAME_WINDOW = st.image([])

    # ── Coluna direita: tradução e métricas ─────────────────────────────────
    with col_info:
        st.markdown('<div class="section-title">🤟 Tradução</div>',
                    unsafe_allow_html=True)

        # Bubble de tradução
        sinal_atual = st.session_state.get("sinal_atual", "AGUARDANDO...")
        st.markdown(f"""
            <div class="translation-label">Sinal Detectado</div>
            <div class="translation-bubble">{sinal_atual}</div>
        """, unsafe_allow_html=True)

        # Métricas
        m1, m2 = st.columns(2)
        with m1:
            st.metric("FPS", "30 Hz")
        with m2:
            confianca = st.session_state.get("confianca", 0.0)
            st.metric("Confiança", f"{confianca:.0%}")

        st.markdown('<div class="gradient-line"></div>', unsafe_allow_html=True)

        # Botão de áudio
        st.markdown('<div class="section-title">🔊 Voz Sintetizada</div>',
                    unsafe_allow_html=True)
        if st.button("Ouvir Tradução"):
            play_audio_non_blocking(
                st.session_state.get("sinal_atual", "Seja bem-vindo ao SINAIS")
            )

        # Hands detectadas
        n_hands = st.session_state.get("n_hands", 0)
        st.markdown(f"""
            <div style="
                margin-top: 16px;
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 14px;
                padding: 14px 18px;
                font-family: 'Satoshi', sans-serif;
                font-size: 0.82rem;
                color: rgba(255,255,255,0.45);
            ">
                🖐️ Mãos detectadas: <span style="color:#bef264; font-weight:700;">
                    {n_hands}
                </span>
            </div>
        """, unsafe_allow_html=True)

    # ── Loop da câmera ───────────────────────────────────────────────────────
    if run_camera:
        mp_hands = mp.solutions.hands
        mp_draw  = mp.solutions.drawing_utils

        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Câmera não encontrada. Verifique a conexão.")
            return

        while cap.isOpened() and run_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Falha na captura do frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = hands.process(frame_rgb)

            n_hands = 0
            if results.multi_hand_landmarks:
                n_hands = len(results.multi_hand_landmarks)
                for hand_lm in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame_rgb, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(106, 72, 242), thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=(190, 242, 100), thickness=2, circle_radius=2),
                    )
                    # Features extraídas prontas para buffer da LSTM
                    _features = extract_and_normalize_landmarks(hand_lm)

            st.session_state["n_hands"] = n_hands
            FRAME_WINDOW.image(frame_rgb, channels="RGB")

        cap.release()


if __name__ == "__main__":
    main()
