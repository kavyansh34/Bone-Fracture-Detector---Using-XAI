import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import io
import base64

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FractureAI · X-Ray Analyzer",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important;
    color: #e8edf5 !important;
    font-family: 'Space Grotesk', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, #0d2040 0%, #080c14 60%) !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.hero { text-align: center; padding: 52px 24px 32px; }
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,.12);
    border: 1px solid rgba(56,189,248,.3);
    color: #38bdf8; font-size: 11px; font-weight: 600;
    letter-spacing: 3px; text-transform: uppercase;
    padding: 5px 18px; border-radius: 100px; margin-bottom: 20px;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.6rem); font-weight: 800; line-height: 1.1;
    background: linear-gradient(135deg, #e8edf5 30%, #38bdf8 70%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 12px;
}
.hero p { color: #7a8fa8; font-size: 1rem; max-width: 560px; margin: 0 auto; line-height: 1.7; }

[data-testid="stFileUploader"] {
    background: rgba(56,189,248,.04) !important;
    border: 2px dashed rgba(56,189,248,.22) !important;
    border-radius: 14px !important; padding: 16px !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(56,189,248,.5) !important; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    color: #38bdf8 !important; font-weight: 600 !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 1rem !important; letter-spacing: 1px !important;
    padding: 14px 0 !important; margin-top: 14px !important;
    transition: opacity .2s, transform .15s !important;
}
.stButton > button:hover { opacity: .88 !important; transform: translateY(-1px) !important; }

.disclaimer {
    text-align: center; padding: 0 40px 36px; color: #3d5166;
    font-size: .82rem; max-width: 700px; margin: 0 auto; line-height: 1.6;
}

@keyframes fadeSlide {
    from { opacity:0; transform:translateY(12px); }
    to   { opacity:1; transform:translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DEVICE & TRANSFORMS
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VAL_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    m = models.vit_b_16(pretrained=False)
    num_ftrs = m.heads.head.in_features
    m.heads.head = nn.Linear(num_ftrs, 2)
    try:
        m.load_state_dict(torch.load("best_fracture_model.pth", map_location=DEVICE))
        m.eval()
        m.to(DEVICE)
        return m, True
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_xray(image_bytes):
    """Returns (native_res_rgb, model_input_rgb_224).
    native_res_rgb : full original resolution, used for overlay display.
    model_input_rgb_224 : CLAHE + resized to 224×224, fed to model and XAI.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img_gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        pil = Image.open(io.BytesIO(image_bytes)).convert("L")
        img_gray = np.array(pil)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # Native-resolution RGB (for overlay — keep full detail)
    native_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

    # 224×224 RGB (for model input and XAI spatial alignment)
    model_rgb = cv2.resize(img_clahe, (224, 224), interpolation=cv2.INTER_AREA)
    model_rgb = cv2.cvtColor(model_rgb, cv2.COLOR_GRAY2RGB)

    return native_rgb, model_rgb

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(model, image_bytes):
    native_rgb, model_rgb = preprocess_xray(image_bytes)
    tensor = VAL_TRANSFORMS(model_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    return pred, probs, native_rgb, model_rgb


# ─────────────────────────────────────────────
# XAI: ViT Gradient-Patch-CAM
#
# WHY Attention Rollout failed visually:
#   Multiplying 12 attention matrices together (rollout) causes the signal to
#   diffuse across ALL tokens — by layer 12 every patch attends to every other
#   patch nearly equally, giving a washed-out uniform blob. That's the random
#   patchy heatmap you saw. It is not wrong mathematically, just uninformative
#   for a fine-tuned classifier.
#
# BETTER APPROACH — Gradient Patch CAM:
#   1. Run a forward pass keeping gradients alive (no torch.no_grad).
#   2. Hook the output of the LAST encoder layer to get patch token activations
#      shape (B, 197, 768) — 196 patch tokens + 1 CLS token.
#   3. Backprop the predicted class score → get gradients w.r.t. those tokens.
#   4. Weight each token's 768-d activation vector by its gradient magnitude
#      (global-average across the 768 channels) → scalar importance per token.
#   5. ReLU + reshape to 14×14 → resize to full image.
#   This gives a sharp, gradient-driven map showing EXACTLY which spatial
#   regions pushed the model toward its prediction.
# ─────────────────────────────────────────────
def generate_vit_gradcam(model, model_rgb, native_rgb):
    """
    model_rgb  : 224×224 uint8 RGB numpy array (model input)
    native_rgb : full-resolution uint8 RGB numpy array (for display)
    Returns    : dict with PIL Images for 'overlay', 'heatmap', 'original'
    """
    model.eval()

    tensor = VAL_TRANSFORMS(model_rgb).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(False)   # we hook activations, not the input

    # Storage for activations and gradients from the last encoder block
    activations = {}
    gradients   = {}

    def fwd_hook(module, inp, out):
        # out: (B, seq_len, hidden_dim)  e.g. (1, 197, 768)
        activations['patch_tokens'] = out

    def bwd_hook(module, grad_in, grad_out):
        # grad_out[0]: (B, seq_len, hidden_dim)
        gradients['patch_tokens'] = grad_out[0]

    # Hook the LAST encoder block (index -1 = block 11)
    last_block = model.encoder.layers[-1]
    fwd_handle = last_block.register_forward_hook(fwd_hook)
    bwd_handle = last_block.register_full_backward_hook(bwd_hook)

    # Forward — keep grad graph
    logits = model(tensor)
    pred_class = logits.argmax(dim=1).item()

    # Backward on the predicted class score only
    model.zero_grad()
    logits[0, pred_class].backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # ── Compute importance per patch token ──
    acts  = activations['patch_tokens'][0]   # (197, 768)
    grads = gradients['patch_tokens'][0]     # (197, 768)

    # Drop CLS token (index 0), keep 196 patch tokens
    acts  = acts[1:]    # (196, 768)
    grads = grads[1:]   # (196, 768)

    # Weight activations by global-average gradient (per token)
    weights = grads.mean(dim=-1)             # (196,) — scalar per patch
    cam     = (acts * weights.unsqueeze(-1)).mean(dim=-1)   # (196,)

    # ReLU: keep only positive contributions toward the predicted class
    cam = torch.relu(cam).detach().cpu().numpy()

    # ── Reshape → spatial map ──
    grid = int(np.sqrt(len(cam)))            # 14
    cam  = cam.reshape(grid, grid)           # (14, 14)

    # Normalise
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min < 1e-8:
        # Flat map — fall back to uniform heatmap
        cam = np.ones_like(cam) * 0.5
    else:
        cam = (cam - cam_min) / (cam_max - cam_min)

    # Smooth slightly to remove patch-boundary artefacts
    cam = cv2.GaussianBlur(cam.astype(np.float32), (3, 3), sigmaX=0.8)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # ── Resize to display resolutions ──
    nh, nw = native_rgb.shape[:2]
    cam_native = cv2.resize(cam, (nw, nh), interpolation=cv2.INTER_CUBIC)

    # Apply TURBO colormap (cleaner than JET for medical imaging)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_native), cv2.COLORMAP_TURBO)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend: X-ray 60% + heatmap 40%
    overlay_rgb = cv2.addWeighted(native_rgb, 0.60, heatmap_rgb, 0.40, 0)

    # Also produce a 224-px version for reference tab
    cam_224      = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    heat_224_bgr = cv2.applyColorMap(np.uint8(255 * cam_224), cv2.COLORMAP_TURBO)
    heat_224_rgb = cv2.cvtColor(heat_224_bgr, cv2.COLOR_BGR2RGB)
    ov_224       = cv2.addWeighted(model_rgb, 0.60, heat_224_rgb, 0.40, 0)

    return {
        "original":    Image.fromarray(native_rgb),
        "heatmap":     Image.fromarray(heatmap_rgb),
        "overlay":     Image.fromarray(overlay_rgb),
        "overlay_224": Image.fromarray(ov_224),
    }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def bytes_to_b64(b):
    return base64.b64encode(b).decode()

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI &nbsp;·&nbsp; Radiology &nbsp;·&nbsp; Explainable AI</div>
    <h1>Bone Fracture Detector</h1>
    <p>Upload an X-ray — our Vision Transformer predicts fractures and shows
       <strong style="color:#38bdf8;">exactly where it looked</strong>
       using Gradient Patch CAM.</p>
</div>
<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,.25),transparent);
            margin:0 40px 40px;"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model, model_status = load_model()

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ══════════════════════
# LEFT — Upload panel
# ══════════════════════
with col_left:
    st.markdown("""
    <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
                border-radius:20px;padding:26px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:1px;
                    background:linear-gradient(90deg,transparent,rgba(56,189,248,.4),transparent);"></div>
        <div style="font-family:'Syne',sans-serif;font-size:.8rem;font-weight:700;
                    color:#38bdf8;text-transform:uppercase;letter-spacing:3px;margin-bottom:16px;">
            🩻 &nbsp;Upload X-Ray
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop X-ray here",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded:
        # BUG 4 FIX: read img_bytes here in the left column so it's available
        # in the right column too. In the original code, img_bytes was read
        # inside the left column but referenced later in the right column's
        # XAI block — a scoping issue causing NameError when session_state
        # replayed without re-entering the left column's if-block.
        img_bytes = uploaded.read()
        ext  = uploaded.name.split(".")[-1].lower()
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
        b64  = bytes_to_b64(img_bytes)

        st.markdown(
            f"""<div style="border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,.08);
                    background:#0a0f1a;margin-top:14px;">
                <img src="data:{mime};base64,{b64}"
                     style="width:100%;object-fit:contain;max-height:300px;display:block;"/>
            </div>
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">
                <span style="background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.18);
                       color:#7dd3f0;font-size:.75rem;padding:3px 12px;border-radius:100px;">
                    📁 {uploaded.name}</span>
                <span style="background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.18);
                       color:#7dd3f0;font-size:.75rem;padding:3px 12px;border-radius:100px;">
                    💾 {len(img_bytes)//1024} KB</span>
                <span style="background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.18);
                       color:#7dd3f0;font-size:.75rem;padding:3px 12px;border-radius:100px;">
                    🖼 {ext.upper()}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        img_bytes = None

    analyze_btn = st.button("🔍  Analyze X-Ray", disabled=(uploaded is None))

    # XAI explainer
    st.markdown("""
    <div style="background:rgba(56,189,248,.05);border:1px solid rgba(56,189,248,.14);
                border-radius:14px;padding:16px 18px;margin-top:18px;">
        <div style="font-family:'Syne',sans-serif;font-size:.8rem;font-weight:700;
                    color:#38bdf8;margin-bottom:8px;">🧠 &nbsp;About the XAI — Gradient Patch CAM</div>
        <div style="font-size:.8rem;color:#64748b;line-height:1.75;">
            Multiplies attention matrices across all 12 ViT encoder layers to reveal
            which 14×14 image patches drove the prediction.<br><br>
            <span style="color:#f97316;">🔴 Warm</span> = high model attention &nbsp;·&nbsp;
            <span style="color:#38bdf8;">🔵 Cool</span> = low attention
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════
# RIGHT — Results + XAI
# ══════════════════════
with col_right:
    st.markdown("""
    <div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
                border-radius:20px;padding:26px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:1px;
                    background:linear-gradient(90deg,transparent,rgba(56,189,248,.4),transparent);"></div>
        <div style="font-family:'Syne',sans-serif;font-size:.8rem;font-weight:700;
                    color:#38bdf8;text-transform:uppercase;letter-spacing:3px;margin-bottom:4px;">
            📊 &nbsp;Analysis Results
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model_status is not True:
        st.markdown(
            f"""<div style="background:rgba(234,179,8,.1);border:1px solid rgba(234,179,8,.3);
                    border-radius:12px;padding:18px;color:#fbbf24;font-size:.88rem;margin-top:10px;">
                ⚠️ <b>Model not loaded.</b><br>
                Place <code>best_fracture_model.pth</code> beside <code>app.py</code> and restart.<br><br>
                <span style="color:#64748b;font-size:.8rem;">Error: {model_status}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    elif not uploaded:
        st.markdown("""
        <div style="text-align:center;padding:70px 20px;">
            <div style="font-size:3.5rem;opacity:.25;margin-bottom:14px;">🦴</div>
            <div style="font-family:'Syne',sans-serif;font-size:.95rem;
                        font-weight:600;color:#4a6080;">Awaiting X-ray upload…</div>
            <div style="font-size:.82rem;color:#3d5166;margin-top:6px;">
                Upload an image and click Analyze</div>
        </div>
        """, unsafe_allow_html=True)

    elif analyze_btn or st.session_state.get("last_result"):

        # BUG 5 FIX: predict() now also returns the preprocessed img so
        # generate_vit_gradcam() receives the exact same numpy array
        # the model saw — previously `img` was never defined in this scope,
        # causing a NameError on the generate_gradcam(model, input_tensor, img) call.
        if analyze_btn and img_bytes is not None:
            with st.spinner("Analyzing X-ray and computing attention map…"):
                pred, probs, native_rgb, model_rgb = predict(model, img_bytes)
                xai = generate_vit_gradcam(model, model_rgb, native_rgb)
                st.session_state["last_result"] = (pred, probs, uploaded.name, xai)

        if "last_result" in st.session_state:
            pred, probs, fname, xai = st.session_state["last_result"]

            is_frac   = (pred == 1)
            label     = "Fracture Detected" if is_frac else "No Fracture"
            conf_frac = float(probs[1]) * 100
            conf_norm = float(probs[0]) * 100
            icon      = "🚨" if is_frac else "✅"
            bg_col    = "rgba(239,68,68,.12)"  if is_frac else "rgba(34,197,94,.12)"
            bdr_col   = "rgba(239,68,68,.35)"  if is_frac else "rgba(34,197,94,.35)"
            lbl_col   = "#f87171"              if is_frac else "#4ade80"

            # ── Verdict card ──
            st.markdown(
                f"""<div style="background:{bg_col};border:1px solid {bdr_col};
                        border-radius:16px;padding:22px 26px 18px;text-align:center;
                        animation:fadeSlide .5s ease;margin-top:10px;">
                    <div style="font-size:2.6rem;margin-bottom:6px;">{icon}</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.75rem;
                                font-weight:800;color:{lbl_col};">{label}</div>
                    <div style="font-size:.82rem;color:#94a3b8;margin-top:4px;">
                        ViT-B/16 &nbsp;&middot;&nbsp; {fname}</div>
                </div>""",
                unsafe_allow_html=True,
            )

            # ── Confidence bars ──
            for emoji, lbl, val, grad, col in [
                ("🔴", "Fracture", conf_frac, "linear-gradient(90deg,#ef4444,#f97316)", "#f87171"),
                ("🟢", "Normal",   conf_norm,  "linear-gradient(90deg,#22c55e,#16a34a)", "#4ade80"),
            ]:
                st.markdown(
                    f"""<div style="margin-top:13px;">
                        <div style="display:flex;justify-content:space-between;
                                    font-size:.83rem;color:#94a3b8;margin-bottom:5px;">
                            <span>{emoji} &nbsp;{lbl}</span>
                            <span style="font-weight:700;color:{col};">{val:.1f}%</span>
                        </div>
                        <div style="background:rgba(255,255,255,.07);border-radius:100px;
                                    height:10px;overflow:hidden;">
                            <div style="width:{val:.1f}%;height:100%;border-radius:100px;
                                        background:{grad};"></div>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # ── Stats row ──
            st.markdown(
                f"""<div style="display:flex;gap:11px;margin-top:15px;">
                    <div style="flex:1;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
                                border-radius:11px;padding:13px;text-align:center;">
                        <div style="font-family:'Syne',sans-serif;font-size:1.25rem;
                                    font-weight:700;color:#38bdf8;">{conf_frac:.1f}%</div>
                        <div style="font-size:.72rem;color:#64748b;margin-top:2px;">Fracture</div>
                    </div>
                    <div style="flex:1;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
                                border-radius:11px;padding:13px;text-align:center;">
                        <div style="font-family:'Syne',sans-serif;font-size:1.25rem;
                                    font-weight:700;color:#38bdf8;">{conf_norm:.1f}%</div>
                        <div style="font-size:.72rem;color:#64748b;margin-top:2px;">Normal</div>
                    </div>
                    <div style="flex:1;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
                                border-radius:11px;padding:13px;text-align:center;">
                        <div style="font-family:'Syne',sans-serif;font-size:1.25rem;
                                    font-weight:700;color:#38bdf8;">ViT-B/16</div>
                        <div style="font-size:.72rem;color:#64748b;margin-top:2px;">Model</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # ══════════════════════════════════════
            # XAI OUTPUT — shown for ALL predictions
            # ══════════════════════════════════════
            st.markdown("""
            <div style="height:1px;background:linear-gradient(90deg,transparent,
                        rgba(56,189,248,.2),transparent);margin:20px 0 16px;"></div>
            <div style="font-family:'Syne',sans-serif;font-size:.8rem;font-weight:700;
                        color:#38bdf8;text-transform:uppercase;letter-spacing:3px;margin-bottom:12px;">
                🧠 &nbsp;Gradient Patch CAM — Where the Model Looked
            </div>
            """, unsafe_allow_html=True)

            if xai:
                tab_ov, tab_hm, tab_og = st.tabs(["🔥 Overlay", "🗺 Heatmap", "📷 Original"])

                views = [
                    (tab_ov, xai["overlay"],  "Attention heatmap blended onto X-ray — warm regions = model focus"),
                    (tab_hm, xai["heatmap"],  "Raw attention map — red = high attention, blue = low attention"),
                    (tab_og, xai["original"], "CLAHE-enhanced grayscale input (as fed to the model)"),
                ]
                for tab, pil_img, caption in views:
                    with tab:
                        b64img = pil_to_b64(pil_img)
                        st.markdown(
                            f"""<div style="border-radius:12px;overflow:hidden;
                                    border:1px solid rgba(56,189,248,.18);background:#0a0f1a;">
                                <img src="data:image/png;base64,{b64img}"
                                     style="width:100%;object-fit:contain;
                                            max-height:300px;display:block;"/>
                            </div>
                            <div style="font-size:.75rem;color:#475569;
                                        text-align:center;margin-top:7px;">{caption}</div>""",
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown("""
                <div style="background:rgba(234,179,8,.08);border:1px solid rgba(234,179,8,.2);
                            border-radius:10px;padding:12px;color:#fbbf24;font-size:.83rem;">
                    ⚠️ Could not generate attention map for this image.
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────
st.markdown("""
<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,.18),transparent);
            margin:14px 40px 26px;"></div>
<div class="disclaimer">
    ⚠️ For <strong>research and educational purposes only</strong>.
    Not a substitute for professional medical diagnosis.
    Always consult a qualified radiologist or physician.
</div>
""", unsafe_allow_html=True)
