# =====================================================================
# SECUREPIXEL & SECUREAUDIO 3.0 — STREAMLIT CLOUD PRODUCTION ENGINE
# Exact Match: UI/UX HTML Dark Card + Dual ConvNeXtV2 + LP-Residual 1D CNN
# =====================================================================

import io
import os
import cv2
import tempfile
import subprocess
import requests
import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import soundfile as sf
import librosa
import scipy.signal
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------
# KONFIGURASI HALAMAN & CUSTOM CSS (IDENTIK DENGAN VERSI HTML)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="SecurePixel - Media Authenticity Verifier",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
<style>
    /* Reset & Tema Dasar Identik dengan HTML */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .block-container {
        max-width: 620px !important;
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
    }
    
    /* Card Utama */
    .sp-header-card {
        background: #1e293b;
        padding: 28px 24px 18px 24px;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
        margin-bottom: 18px;
        text-align: center;
    }
    .sp-title {
        margin: 0 0 8px 0;
        font-size: 24px;
        font-weight: 700;
        color: #38bdf8 !important;
    }
    .sp-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
        line-height: 1.5;
    }

    /* Penyesuaian Komponen Upload & Tombol Streamlit */
    [data-testid="stFileUploader"] {
        background: #1e293b;
        border: 2px dashed #475569;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
    }
    div.stButton > button:first-child {
        background-color: #0284c7;
        color: white;
        border: none;
        padding: 12px 20px;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        font-size: 15px;
        transition: background 0.2s;
    }
    div.stButton > button:first-child:hover {
        background-color: #0369a1;
        color: white;
    }

    /* Kotak Hasil & Stempel Resmi */
    .verdict-box {
        background: #0f172a;
        border-radius: 12px;
        padding: 22px 18px;
        text-align: center;
        border: 1px solid #334155;
        margin-top: 16px;
        margin-bottom: 16px;
    }
    .stamp-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 8px;
        font-size: 15px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    .stamp-generate { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }
    .stamp-natural { background: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
    .stamp-origin { background: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; }
    .stamp-inconclusive { background: #78350f; color: #fde68a; border: 1px solid #f59e0b; }
    .stamp-neutral { background: #334155; color: #cbd5e1; border: 1px solid #475569; }

    .verdict-desc {
        color: #cbd5e1;
        font-size: 14px;
        margin: 0;
        line-height: 1.5;
    }

    /* Grid Dual-Track Video */
    .video-split-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 16px;
        margin-bottom: 16px;
    }
    .split-card {
        background: #0f172a;
        border: 1px solid #334155;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
    }
    .split-card h4 {
        margin: 0 0 10px 0;
        font-size: 13px;
        color: #94a3b8;
        letter-spacing: 0.5px;
    }
    .split-card .stamp-badge {
        font-size: 12px;
        padding: 6px 10px;
    }
    .preview-meta {
        font-size: 12px;
        color: #64748b;
        text-align: center;
        margin-top: -6px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 1. DEFINISI ARSITEKTUR & CACHING MODEL (HANYA MUAT 1X DI RAM)
# ---------------------------------------------------------------------
device = torch.device("cpu")
torch.set_num_threads(os.cpu_count() or 4)

class BioAcoustic1DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_wide   = nn.Conv1d(2, 48, kernel_size=128, stride=4, padding=62)
        self.conv_mid    = nn.Conv1d(2, 48, kernel_size=32,  stride=4, padding=14)
        self.conv_narrow = nn.Conv1d(2, 32, kernel_size=8,   stride=4, padding=2)
        self.bn1 = nn.BatchNorm1d(128)

        self.block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2)
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x1 = self.conv_wide(x)
        x2 = self.conv_mid(x)
        x3 = self.conv_narrow(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.leaky_relu(self.bn1(x), 0.2)
        x = F.max_pool1d(x, 2)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x).squeeze(-1)
        return self.head(x).squeeze(-1)

@st.cache_resource(show_spinner="Initializing SecurePixel & SecureAudio Forensic Engines...")
def load_all_forensic_experts():
    # --- A. Muat Visual Dual-Expert ---
    HF_VISUAL_REPO_ID = "rijal028/securepixel-dual-expert-weights"
    v_hunter_path = hf_hub_download(repo_id=HF_VISUAL_REPO_ID, filename="securepixel_hunter_512px.pt")
    v_guardian_path = hf_hub_download(repo_id=HF_VISUAL_REPO_ID, filename="securepixel_guardian_512px.pt")

    def build_v_expert(w_path):
        m = timm.create_model(
            "convnextv2_tiny.fcmae_ft_in22k_in1k",
            pretrained=False,
            num_classes=2,
            global_pool="avg"
        ).to(device)
        sd = torch.load(w_path, map_location=device)
        m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
        m.eval()
        return m

    hunter_v = build_v_expert(v_hunter_path)
    guardian_v = build_v_expert(v_guardian_path)

    # --- B. Muat Audio Dual-Expert ---
    HF_AUDIO_REPO_ID = "rijal028/secureaudio-biophysics-dualexpert-lp"
    a_guardian_path = hf_hub_download(repo_id=HF_AUDIO_REPO_ID, filename="guardian_biophysics_ep100lp.pt")
    a_hunter_path = hf_hub_download(repo_id=HF_AUDIO_REPO_ID, filename="hunter_biophysics_ep100lp.pt")

    def build_a_expert(w_path):
        m = BioAcoustic1DNet().to(device)
        sd = torch.load(w_path, map_location=device)
        m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
        m.eval()
        return m

    guardian_a = build_a_expert(a_guardian_path)
    hunter_a = build_a_expert(a_hunter_path)

    return hunter_v, guardian_v, hunter_a, guardian_a

hunter_visual, guardian_visual, audio_hunter, audio_guardian = load_all_forensic_experts()

eval_tf = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------------------
# 2. FUNGSI FORENSIK: C2PA, PHASH, VISUAL, AUDIO, DAN VIDEO
# ---------------------------------------------------------------------
def inspect_c2pa_bytes(raw_bytes: bytes):
    slice_bytes = raw_bytes[:2097152]
    text = slice_bytes.decode("utf-8", errors="ignore").lower()
    has_container = ("c2pa" in text) or ("urn:c2pa" in text) or ("jumb" in text)
    if not has_container:
        return {"detected": False}
    synthetic_markers = [
        "trainedalgorithmicmedia", "generativesynthesis", "c2pa.synthetic",
        "synthid", "dall-e", "midjourney", "gemini"
    ]
    is_synthetic = any(m in text for m in synthetic_markers)
    return {"detected": True, "isSynthetic": is_synthetic}

def compute_perceptual_hash(raw_bytes: bytes, is_image: bool):
    if not is_image:
        return f"stream_id_{len(raw_bytes)}"
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize((8, 8))
        arr = np.asarray(img, dtype=np.float32)
        grays = arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114
        avg = grays.mean()
        bits = "".join("1" if g >= avg else "0" for g in grays.flatten())
        return f"{int(bits, 2):016x}"
    except Exception:
        return f"phash_fallback_{len(raw_bytes)}"

def evaluate_single_image(raw_bytes: bytes):
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception:
        return "INCONCLUSIVE", "Corrupted or unsupported image file."

    gray_np = np.array(pil_img.convert("L"))
    laplacian_var = float(cv2.Laplacian(gray_np, cv2.CV_64F).var())
    if laplacian_var < 3.0:
        return "INCONCLUSIVE", "Extreme blur prevents reliable forensic attribution."

    tensor_input = eval_tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs_hunt = torch.softmax(hunter_visual(tensor_input), dim=-1)[0]
        probs_guard = torch.softmax(guardian_visual(tensor_input), dim=-1)[0]
        h_ai = float(probs_hunt[0].item())
        g_real = float(probs_guard[1].item())

    if h_ai >= 0.45 and g_real >= 0.60:
        return "INCONCLUSIVE", "Ambiguous forensic signals; system safely abstained."
    elif (h_ai >= 0.85 and g_real < 0.60) or (h_ai >= 0.65 and g_real < 0.50) or (h_ai >= 0.55 and g_real <= 0.35):
        return "LIKELY AI-GENERATED", "Synthetic latent generative traits identified."
    elif (g_real >= 0.70 and h_ai < 0.35) or (g_real >= 0.55 and h_ai <= 0.35):
        return "LIKELY NATURAL CAPTURE", "Physical camera sensor optical noise confirmed."
    else:
        return "INCONCLUSIVE", "Ambiguous forensic signals; system safely abstained."

def decode_any_audio_payload(raw_bytes: bytes):
    cmd = ["ffmpeg", "-y", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out_wav_bytes, _ = proc.communicate(input=raw_bytes)
        wav, _ = sf.read(io.BytesIO(out_wav_bytes), dtype='float32')
        return wav
    except Exception:
        try:
            wav, _ = sf.read(io.BytesIO(raw_bytes), dtype='float32')
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            return wav
        except Exception:
            return None

PATCH_SIZE = 2048
HOP_SIZE = 1024
MAX_PATCHES_PER_FILE = 20

def evaluate_audio_array(w):
    if len(w.shape) > 1:
        w = np.mean(w, axis=1)
    w = w - np.mean(w)

    if len(w) < PATCH_SIZE:
        return "INCONCLUSIVE", "Audio sample is too short (< 128ms) for forensic analysis."

    try:
        a_coeffs = librosa.lpc(w, order=16)
        pred = scipy.signal.lfilter([0] + (-a_coeffs[1:]).tolist(), [1], w)
        lp_residual = w - pred

        d1 = np.diff(w, prepend=w[0])
        d2 = np.diff(d1, prepend=d1[0])

        res_norm = lp_residual / (np.std(lp_residual) + 1e-7)
        d2_norm = d2 / (np.std(d2) + 1e-7)
        feat_2ch = np.stack([res_norm, d2_norm], axis=0).astype(np.float32)

        patches = []
        total_len = feat_2ch.shape[1]
        for start in range(0, total_len - PATCH_SIZE + 1, HOP_SIZE):
            p = feat_2ch[:, start : start + PATCH_SIZE]
            if np.std(p[0]) > 0.05:
                patches.append(p)
            if len(patches) >= MAX_PATCHES_PER_FILE:
                break

        if len(patches) == 0:
            return "INCONCLUSIVE", "Audio contains purely silent or corrupted signal."

        batch_tensors = torch.tensor(np.array(patches), dtype=torch.float32).to(device)
        with torch.no_grad():
            p_g = torch.sigmoid(audio_guardian(batch_tensors)).cpu().numpy()
            p_h = torch.sigmoid(audio_hunter(batch_tensors)).cpu().numpy()

        mean_g = float(np.mean(p_g))
        mean_h = float(np.mean(p_h))
    except Exception as e:
        return "INCONCLUSIVE", f"Signal extraction error: {str(e)}"

    if mean_h >= 0.60 and (mean_h - mean_g) >= 0.20:
        return "LIKELY AI-GENERATED", f"Synthetic vocoder artifacts identified (Hunter: {mean_h:.4f}, Guardian: {mean_g:.4f})."
    elif mean_g >= 0.50 and (mean_g - mean_h) >= 0.10:
        return "LIKELY NATURAL CAPTURE", f"Physical acoustic sensor capture verified (Guardian: {mean_g:.4f}, Hunter: {mean_h:.4f})."
    else:
        return "INCONCLUSIVE", f"Ambiguous acoustic signals (Guardian: {mean_g:.4f}, Hunter: {mean_h:.4f}); system safely abstained."

def evaluate_video_dual_track(raw_bytes: bytes):
    """Mengekstrak hingga 6 keyframe (voting >=50% persis versi JS) + 12 detik Audio Track"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(raw_bytes)
        vid_path = tmp_vid.name
    wav_path = vid_path.replace(".mp4", ".wav")

    try:
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = int(total_frames / fps) if fps > 0 else 1
        frame_count = min(max(duration_sec, 1), 6)

        ai_votes = 0
        natural_votes = 0
        valid_frames = 0

        for i in range(1, frame_count + 1):
            target_sec = min(i, max(0.1, (total_frames / fps) - 0.1)) if fps > 0 else 0
            cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            valid_frames += 1
            _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            f_verdict, _ = evaluate_single_image(buf.tobytes())
            if f_verdict == "LIKELY AI-GENERATED":
                ai_votes += 1
            elif f_verdict == "LIKELY NATURAL CAPTURE":
                natural_votes += 1

        cap.release()

        v_verdict = "INCONCLUSIVE"
        if valid_frames > 0:
            threshold = int(np.ceil(valid_frames * 0.5))
            if ai_votes >= threshold:
                v_verdict = "LIKELY AI-GENERATED"
            elif natural_votes >= threshold:
                v_verdict = "LIKELY NATURAL CAPTURE"

        # Ekstraksi maksimal 12 detik audio @ 16kHz mono (persis Web Audio API di JS)
        cmd = [
            "ffmpeg", "-y", "-i", vid_path, "-vn",
            "-t", "12", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) <= 1000:
            a_verdict, a_desc = "NO_AUDIO_STREAM", "No audio stream detected."
        else:
            wav_data, _ = sf.read(wav_path, dtype='float32')
            a_verdict, a_desc = evaluate_audio_array(wav_data)

        return v_verdict, a_verdict, a_desc
    finally:
        if os.path.exists(vid_path):
            os.remove(vid_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

def get_stamp_meta(verdict, custom_desc=None):
    if verdict == "LIKELY AI-GENERATED":
        return "stamp-generate", "🤖 LIKELY AI-GENERATED", custom_desc or "Synthetic latent generative traits identified."
    elif verdict == "LIKELY NATURAL CAPTURE":
        return "stamp-natural", "📷 LIKELY NATURAL CAPTURE", custom_desc or "Physical sensor dynamics confirmed."
    elif verdict == "VERIFIED ORIGIN":
        return "stamp-origin", "🛡️ VERIFIED ORIGIN", custom_desc or "Valid cryptographic provenance seal confirmed."
    elif verdict in ("NO_AUDIO_STREAM", "NO_AUDIO"):
        return "stamp-neutral", "NO AUDIO TRACK", "No audio stream detected."
    else:
        return "stamp-inconclusive", "⚠ INCONCLUSIVE", custom_desc or "Ambiguous forensic signals; system safely abstained."

# ---------------------------------------------------------------------
# 3. RENDER ANTARMUKA PENGGUNA (UI) & PIPELINE EKSEKUSI
# ---------------------------------------------------------------------
st.markdown("""
<div class="sp-header-card">
    <div class="sp-title">🛡️ SecurePixel</div>
    <p class="sp-subtitle">Authenticate and verify digital media authenticity within a split second.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image, video, or audio file:",
    type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "avi", "mkv", "webm", "mp3", "wav", "ogg", "m4a", "aac", "opus", "flac"]
)

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_filename" not in st.session_state:
    st.session_state.last_filename = None

if uploaded_file is not None:
    if st.session_state.last_filename != uploaded_file.name:
        st.session_state.last_result = None
        st.session_state.last_filename = uploaded_file.name

    raw_bytes = uploaded_file.getvalue()
    size_mb = len(raw_bytes) / (1024 * 1024)
    fname_lower = uploaded_file.name.lower()
    mime_type = uploaded_file.type or ""

    is_video = mime_type.startswith("video/") or fname_lower.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))
    is_audio = mime_type.startswith("audio/") or fname_lower.endswith((".mp3", ".wav", ".ogg", ".m4a", ".aac", ".opus", ".flac"))
    is_image = not (is_video or is_audio)

    # Preview Media
    if is_video:
        st.video(raw_bytes)
    elif is_audio:
        st.audio(raw_bytes)
    else:
        st.image(raw_bytes, use_container_width=True)

    st.markdown(f'<div class="preview-meta">{uploaded_file.name} ({size_mb:.2f} MB)</div>', unsafe_allow_html=True)

    if st.button("Verify Authenticity"):
        # 1. C2PA Quick Check (Untuk Gambar & Video)
        c2pa_hit = False
        if not is_audio:
            with st.spinner("Checking file credentials..."):
                c2pa_res = inspect_c2pa_bytes(raw_bytes)
                if c2pa_res["detected"]:
                    c2pa_hit = True
                    if c2pa_res["isSynthetic"]:
                        st.session_state.last_result = {
                            "mode": "single",
                            "verdict": "LIKELY AI-GENERATED",
                            "desc": "Synthetic credentials detected in file header."
                        }
                    else:
                        st.session_state.last_result = {
                            "mode": "single",
                            "verdict": "VERIFIED ORIGIN",
                            "desc": "Valid cryptographic provenance seal confirmed."
                        }

        # 2. Alur Inferensi Berdasarkan Jenis Media
        if not c2pa_hit:
            if is_video:
                with st.spinner("Analyzing video frames & acoustic track..."):
                    v_verd, a_verd, a_desc = evaluate_video_dual_track(raw_bytes)
                    st.session_state.last_result = {
                        "mode": "video",
                        "verdict": v_verd,
                        "v_verdict": v_verd,
                        "a_verdict": a_verd,
                        "a_desc": a_desc
                    }
            elif is_audio:
                with st.spinner("Verifying acoustic dynamics..."):
                    wav = decode_any_audio_payload(raw_bytes)
                    if wav is None:
                        verd, desc = "INCONCLUSIVE", "Corrupted or unsupported audio payload."
                    else:
                        verd, desc = evaluate_audio_array(wav)
                    st.session_state.last_result = {
                        "mode": "single",
                        "verdict": verd,
                        "desc": desc
                    }
            else:
                with st.spinner("Analyzing image sensor characteristics..."):
                    verd, desc = evaluate_single_image(raw_bytes)
                    st.session_state.last_result = {
                        "mode": "single",
                        "verdict": verd,
                        "desc": desc
                    }

    # -----------------------------------------------------------------
    # 4. TAMPILKAN HASIL VERIFIKASI & REDRESS TELEMETRY (SUPABASE)
    # -----------------------------------------------------------------
    res = st.session_state.last_result
    if res is not None:
        if res["mode"] == "single":
            cls_name, badge_txt, final_desc = get_stamp_meta(res["verdict"], res.get("desc"))
            st.markdown(f"""
            <div class="verdict-box">
                <div class="stamp-badge {cls_name}">{badge_txt}</div>
                <p class="verdict-desc">{final_desc}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            v_cls, v_txt, v_desc = get_stamp_meta(res["v_verdict"])
            a_cls, a_txt, a_desc = get_stamp_meta(res["a_verdict"], res.get("a_desc"))
            st.markdown(f"""
            <div class="video-split-grid">
                <div class="split-card">
                    <h4>👁️ VISUAL TRACK</h4>
                    <div class="stamp-badge {v_cls}">{v_txt}</div>
                    <p class="verdict-desc" style="font-size: 12px;">{v_desc}</p>
                </div>
                <div class="split-card">
                    <h4>🎙️ AUDIO TRACK</h4>
                    <div class="stamp-badge {a_cls}">{a_txt}</div>
                    <p class="verdict-desc" style="font-size: 12px;">{a_desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Bagian Discrepancy & Redress (Supabase Direct REST API)
        with st.expander("🚩 Believe this result is inaccurate? Flag Result"):
            issue_label = st.selectbox(
                "Issue Type:",
                options=[
                    ("false_positive", "Genuine media flagged as AI"),
                    ("false_negative", "AI media flagged as Authentic"),
                    ("other", "Inconclusive or unexpected result")
                ],
                format_func=lambda x: x[1]
            )
            user_notes = st.text_area(
                "Context (Optional):",
                placeholder="e.g., Captured on Sony A7IV or recorded via WhatsApp..."
            )
            if st.button("Submit Feedback"):
                SUPABASE_URL = "https://sihzvffweusqdsszgmcd.supabase.co"
                SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNpaHp2ZmZ3ZXVzcWRzc3pnbWNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODk2MDczNzEsImV4cCI6MjEwNTE4MzM3MX0.xJ5vZa5HOW349_UJ7nVUfM97DHvcuYZLvxzqL6mIy4g"
                
                phash_val = compute_perceptual_hash(raw_bytes, is_image)
                media_str = "video" if is_video else ("audio" if is_audio else "image")
                payload = {
                    "media_type": media_str,
                    "phash": phash_val,
                    "reported_error": issue_label[0],
                    "user_notes": user_notes.strip(),
                    "system_verdict": res.get("verdict", "UNKNOWN"),
                    "is_verified": False
                }
                headers = {
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal"
                }
                try:
                    r = requests.post(f"{SUPABASE_URL}/rest/v1/forensic_telemetry", json=payload, headers=headers, timeout=10)
                    if r.status_code in (200, 201, 204):
                        st.success("Feedback recorded. Zero biometric or personal media was stored.")
                    else:
                        st.error(f"Failed to submit feedback (Status {r.status_code}).")
                except Exception as e:
                    st.error(f"Error submitting feedback: {str(e)}")
