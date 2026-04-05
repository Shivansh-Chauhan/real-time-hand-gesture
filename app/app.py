"""
app.py  –  Streamlit Web Interface
───────────────────────────────────
Pages
  🏠 Home           – project overview
  📸 Live Demo      – webcam inference (streamlit-webrtc)
  📊 Analytics      – training curves, confusion matrix, class distribution
  ⚙️  Train          – trigger training from the UI
  ℹ️  About          – architecture docs
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_MODEL_PATH, HISTORY_PATH, LOGS_DIR,
    ALL_GESTURES, ALPHABET_GESTURES, NUMBER_GESTURES, COMMAND_GESTURES,
    CONFIDENCE_THRESHOLD, GESTURE_DISPLAY,
)
from src.feature_extractor import extract_features
from src.hand_tracker      import HandTracker
from src.utils             import PredictionSmoother, FPSCounter, OverlayRenderer


# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Hand Gesture Recognition",
    page_icon  = "🤚",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Dark gradient header */
  .main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }
  .main-header h1 { color: #e2e8f0; font-size: 2.4rem; margin: 0; }
  .main-header p  { color: #94a3b8; font-size: 1rem; margin: 0.4rem 0 0 0; }

  /* Metric cards */
  .metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
  }
  .metric-card .value { font-size: 2.2rem; font-weight: 700; color: #38bdf8; }
  .metric-card .label { font-size: 0.85rem; color: #94a3b8; margin-top: 0.2rem; }

  /* Gesture badge */
  .gesture-badge {
    display: inline-block;
    background: #0f3460;
    color: #7dd3fc;
    border: 1px solid #1e40af;
    border-radius: 8px;
    padding: 0.25rem 0.6rem;
    font-size: 0.82rem;
    margin: 2px;
  }

  /* Prediction box */
  .pred-box {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border: 2px solid #38bdf8;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
  }
  .pred-box .gesture { font-size: 3rem; font-weight: 800; color: #7dd3fc; }
  .pred-box .conf    { font-size: 1rem; color: #94a3b8; }

  /* Sentence display */
  .sentence-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-size: 1.4rem;
    font-family: monospace;
    color: #e2e8f0;
    letter-spacing: 0.05em;
    min-height: 56px;
  }

  /* Pipeline step */
  .step-card {
    background: #1e293b;
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
  }
  .step-card h4 { margin: 0; color: #7dd3fc; }
  .step-card p  { margin: 0.2rem 0 0 0; color: #94a3b8; font-size: 0.88rem; }

  /* Status pill */
  .pill-green { background:#064e3b; color:#6ee7b7; border-radius:20px;
                padding:0.2rem 0.8rem; font-size:0.8rem; }
  .pill-red   { background:#7f1d1d; color:#fca5a5; border-radius:20px;
                padding:0.2rem 0.8rem; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)


# ── cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading gesture model …")
def load_predictor():
    """Load model once and cache across reruns."""
    try:
        from src.gesture_model import GesturePredictor
        return GesturePredictor(), None
    except FileNotFoundError as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def load_history():
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_metrics():
    p = LOGS_DIR / "eval_metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ── sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🤚 Gesture Recognition")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📸 Live Demo", "📊 Analytics", "⚙️ Train", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    model_ok = BEST_MODEL_PATH.exists()
    status_html = (
        '<span class="pill-green">✅ Model ready</span>'
        if model_ok else
        '<span class="pill-red">⚠️ No model</span>'
    )
    st.markdown(f"**Model status:** {status_html}", unsafe_allow_html=True)
    st.markdown(f"**Gesture classes:** {len(ALL_GESTURES)}")
    st.markdown("---")
    st.caption("Built with MediaPipe · TensorFlow · Streamlit")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : HOME
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
      <h1>🤚 Hand Gesture Recognition System</h1>
      <p>Real-time AI-powered sign language & gesture detection using computer vision</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    for col, val, label in [
        (col1, len(ALPHABET_GESTURES), "Alphabet Gestures"),
        (col2, len(NUMBER_GESTURES),   "Number Gestures"),
        (col3, len(COMMAND_GESTURES),  "Command Gestures"),
        (col4, 21,                      "Hand Landmarks"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="value">{val}</div>
              <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Gesture catalogue
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### 🔤 Alphabets (A–Z)")
        badges = " ".join(
            f'<span class="gesture-badge">{g}</span>'
            for g in ALPHABET_GESTURES
        )
        st.markdown(badges, unsafe_allow_html=True)

    with col_b:
        st.markdown("#### 🔢 Numbers (0–9)")
        badges = " ".join(
            f'<span class="gesture-badge">{g}</span>'
            for g in NUMBER_GESTURES
        )
        st.markdown(badges, unsafe_allow_html=True)

    with col_c:
        st.markdown("#### 🖐️ Commands")
        for g in COMMAND_GESTURES:
            label = GESTURE_DISPLAY.get(g, g)
            st.markdown(f'<span class="gesture-badge">{label}</span>',
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔄 System Pipeline")
    steps = [
        ("📷 Capture", "Webcam frame at 30 FPS → OpenCV BGR image"),
        ("🤚 Detect", "MediaPipe Hands → 21 3-D landmarks per hand"),
        ("📐 Extract", "88-D feature vector: normalised coords + angles + distances"),
        ("🧠 Classify", "Deep MLP neural network → softmax over all classes"),
        ("🔄 Smooth", "Majority-vote over last 7 frames to remove flicker"),
        ("🔊 Output", "Overlay on frame + TTS announcement + sentence builder"),
    ]
    for title, desc in steps:
        st.markdown(f"""
        <div class="step-card">
          <h4>{title}</h4>
          <p>{desc}</p>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : LIVE DEMO
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📸 Live Demo":
    st.markdown("## 📸 Live Gesture Recognition")
    st.info(
        "Upload an image or use the webcam snapshot below. "
        "For real-time streaming, run `python run_recognition.py` in a terminal."
    )

    predictor, err = load_predictor()
    if err:
        st.error(f"⚠️ {err}\n\nRun `python train_model.py` first.")
        st.stop()

    # ── session state ─────────────────────────────────────────────────────────
    if "sentence" not in st.session_state:
        st.session_state.sentence = ""
    if "last_gesture" not in st.session_state:
        st.session_state.last_gesture = ""

    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Webcam Snapshot"])

    # ── image upload ──────────────────────────────────────────────────────────
    with tab1:
        uploaded = st.file_uploader(
            "Upload a hand gesture image",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            frame   = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            tracker = HandTracker()
            result  = tracker.process(frame, draw=True)
            tracker.close()

            col_img, col_pred = st.columns([3, 2])
            with col_img:
                rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption="Detected landmarks", use_container_width=True)

            with col_pred:
                if result.primary:
                    feat = extract_features(result.primary.landmarks)
                    if feat is not None:
                        label, conf, _ = predictor.predict(feat)
                        top_k = predictor.top_k(feat, k=5)
                        display = GESTURE_DISPLAY.get(label, label)

                        st.markdown(f"""
                        <div class="pred-box">
                          <div class="gesture">{display}</div>
                          <div class="conf">Confidence: {conf*100:.1f}%</div>
                        </div>""", unsafe_allow_html=True)

                        st.markdown("**Top predictions:**")
                        for lbl, c in top_k:
                            bar_col = "🟢" if c > 0.7 else ("🟡" if c > 0.4 else "🔴")
                            st.progress(float(c), text=f"{bar_col} {lbl}: {c*100:.1f}%")
                    else:
                        st.warning("Could not extract features from landmarks.")
                else:
                    st.warning("No hand detected in the image.")

    # ── sentence builder ──────────────────────────────────────────────────────
    with tab2:
        st.markdown(
            "Enable camera access in your browser, then click **Take Photo**."
        )
        camera_img = st.camera_input("Take a photo of your hand gesture")

        if camera_img:
            img_pil = Image.open(camera_img).convert("RGB")
            frame   = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            tracker = HandTracker()
            result  = tracker.process(frame, draw=True)
            tracker.close()

            c1, c2 = st.columns([3, 2])
            with c1:
                rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(rgb, use_container_width=True)

            with c2:
                if result.primary:
                    feat = extract_features(result.primary.landmarks)
                    if feat is not None:
                        label, conf, _ = predictor.predict(feat)
                        display = GESTURE_DISPLAY.get(label, label)

                        st.markdown(f"""
                        <div class="pred-box">
                          <div class="gesture">{display}</div>
                          <div class="conf">{conf*100:.1f}% confidence</div>
                        </div>""", unsafe_allow_html=True)

                        c_add, c_space, c_clear = st.columns(3)
                        if c_add.button("➕ Add"):
                            if len(label) == 1:
                                st.session_state.sentence += label
                            else:
                                st.session_state.sentence += f" {label} "
                        if c_space.button("␣ Space"):
                            st.session_state.sentence += " "
                        if c_clear.button("🗑 Clear"):
                            st.session_state.sentence = ""

    st.markdown("#### 📝 Sentence Builder")
    st.markdown(
        f'<div class="sentence-box">{st.session_state.sentence or "…"}</div>',
        unsafe_allow_html=True,
    )
    if st.button("📋 Copy to clipboard"):
        st.write(f"`{st.session_state.sentence}`")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Analytics":
    st.markdown("## 📊 Model Analytics")

    history = load_history()
    metrics = load_metrics()

    if history is None and metrics is None:
        st.warning("No training data found. Run `python train_model.py` first.")
        st.stop()

    # Training curves
    if history:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        epochs = list(range(1, len(history["accuracy"]) + 1))
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Accuracy", "Loss"),
        )
        fig.add_trace(go.Scatter(x=epochs, y=history["accuracy"],
                                 name="Train Acc",  line=dict(color="#38bdf8")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history["val_accuracy"],
                                 name="Val Acc",    line=dict(color="#f472b6", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history["loss"],
                                 name="Train Loss", line=dict(color="#34d399")), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
                                 name="Val Loss",   line=dict(color="#fb923c", dash="dash")), row=1, col=2)
        fig.update_layout(
            template="plotly_dark", height=400,
            title="Training History", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Metrics summary
    if metrics:
        col1, col2, col3 = st.columns(3)
        acc = metrics.get("accuracy", 0)
        report = metrics.get("classification_report", {})
        macro_f1  = report.get("macro avg", {}).get("f1-score", 0)
        n_classes = len(metrics.get("class_names", []))

        with col1:
            st.metric("Test Accuracy", f"{acc*100:.2f}%")
        with col2:
            st.metric("Macro F1-Score", f"{macro_f1*100:.2f}%")
        with col3:
            st.metric("Classes", n_classes)

        # Confusion matrix image
        cm_img = LOGS_DIR / "confusion_matrix.png"
        if cm_img.exists():
            st.markdown("#### Confusion Matrix")
            st.image(str(cm_img), use_container_width=True)

        # Per-class table
        if report:
            import pandas as pd
            rows = []
            for cls, vals in report.items():
                if cls in ("accuracy", "macro avg", "weighted avg"):
                    continue
                if isinstance(vals, dict):
                    rows.append({
                        "Gesture"  : cls,
                        "Precision": f"{vals.get('precision', 0)*100:.1f}%",
                        "Recall"   : f"{vals.get('recall', 0)*100:.1f}%",
                        "F1-Score" : f"{vals.get('f1-score', 0)*100:.1f}%",
                        "Support"  : int(vals.get("support", 0)),
                    })
            if rows:
                df = pd.DataFrame(rows)
                st.markdown("#### Per-class Metrics")
                st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : TRAIN
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Train":
    st.markdown("## ⚙️ Train / Retrain Model")
    st.warning(
        "Training runs inside this Streamlit process. "
        "For large datasets, prefer running `python train_model.py` in a terminal."
    )

    with st.form("train_form"):
        st.markdown("#### Hyperparameters")
        c1, c2, c3 = st.columns(3)
        epochs    = c1.number_input("Epochs",         min_value=5, max_value=500, value=100)
        batch_sz  = c2.number_input("Batch size",     min_value=16, max_value=512, value=64)
        lr        = c3.number_input("Learning rate",  min_value=1e-5, max_value=0.1,
                                    value=0.001, format="%.5f")

        augment   = st.checkbox("Data augmentation", value=True)
        aug_fac   = st.slider("Augmentation factor", 1, 10, 3,
                              disabled=not augment)

        submitted = st.form_submit_button("🚀 Start Training")

    if submitted:
        from src.dataset_manager import DatasetManager
        from src.gesture_model   import GestureModelTrainer
        from config import MODEL_CFG

        MODEL_CFG.update({"epochs": epochs, "batch_size": batch_sz, "learning_rate": lr})

        log_area = st.empty()
        prog_bar = st.progress(0)

        with st.spinner("Training in progress …"):
            try:
                dm = DatasetManager()
                X, y = dm.load_raw(augment=augment, augment_factor=aug_fac)
                trainer = GestureModelTrainer()
                X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
                num_classes = len(trainer.label_encoder.classes_)

                trainer.train(X_train, y_train, X_val, y_val, num_classes)
                metrics = trainer.evaluate(X_test, y_test)
                trainer.save()

                st.success(
                    f"✅ Training complete! "
                    f"Test accuracy: **{metrics['accuracy']*100:.2f}%**"
                )
                st.cache_data.clear()
                st.cache_resource.clear()
            except FileNotFoundError:
                st.error(
                    "No dataset found. "
                    "Run `python collect_data.py --all` first."
                )
            except Exception as e:
                st.error(f"Training failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown("## ℹ️ System Architecture")

    st.markdown("""
    ### 📐 Feature Engineering

    Each hand produces a **88-dimensional feature vector**:

    | Segment | Dimension | Description |
    |---------|-----------|-------------|
    | Normalised coords | 63 | 21 landmarks × (x, y, z) — centred at wrist, scaled by hand span |
    | Joint angles | 15 | Bend angles at each finger joint triplet |
    | Inter-landmark distances | 10 | Key fingertip-to-fingertip distances |

    ### 🧠 Model Architecture

    ```
    Input (88) → Dense(512)+BN+ReLU+Dropout → +Skip
               → Dense(256)+BN+ReLU+Dropout → +Skip
               → Dense(128)+BN+ReLU+Dropout → +Skip
               → Dense(64)+BN+ReLU+Dropout  → +Skip
               → Dense(NUM_CLASSES) → Softmax
    ```

    - **Residual skip connections** for gradient flow
    - **BatchNormalization** for training stability
    - **Dropout (40%)** for regularisation
    - **Adam** optimiser with ReduceLROnPlateau
    - **EarlyStopping** with best-weight restoration

    ### 🔄 Inference Pipeline

    1. MediaPipe detects up to **2 hands** per frame
    2. Features extracted from 21 3-D landmarks
    3. StandardScaler normalises features (fitted on training data)
    4. Model predicts softmax probabilities
    5. **Majority-vote smoother** (7-frame window) reduces flicker
    6. Result displayed + spoken via TTS if confidence ≥ 65%

    ### 🛠️ Tech Stack

    | Layer | Library |
    |-------|---------|
    | Hand tracking | MediaPipe 0.10 |
    | Deep learning | TensorFlow / Keras |
    | Computer vision | OpenCV |
    | TTS | pyttsx3 / gTTS |
    | Web UI | Streamlit |
    | Data | NumPy, scikit-learn |
    | Visualisation | Matplotlib, Seaborn, Plotly |
    """)
