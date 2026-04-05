# 🤚 Hand Gesture Recognition System

> Real-time AI-powered hand gesture recognition supporting A–Z alphabets, 0–9 numbers, and 12 custom commands using MediaPipe + TensorFlow + Streamlit.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **38 gesture classes** | A–Z (26) · 0–9 (10) · Commands (12) |
| **21-landmark tracking** | MediaPipe Hands, up to 2 simultaneous hands |
| **88-D feature vectors** | Normalised coords + joint angles + distances |
| **Deep MLP classifier** | Residual MLP, batch-norm, dropout, ~95%+ accuracy |
| **7-frame smoothing** | Majority-vote to eliminate prediction flicker |
| **Text-to-speech** | pyttsx3 (offline) / gTTS (online) |
| **Sentence builder** | Auto-appends recognised characters |
| **Streamlit dashboard** | Live demo, analytics, training UI |
| **Screenshot capture** | One-key save annotated frames |

---

## 🗂 Project Structure

```
gesture_recognition/
│
├── app.py                    ← Streamlit web interface
├── run_recognition.py        ← Real-time OpenCV application
├── collect_data.py           ← Webcam dataset collector
├── train_model.py            ← Full training pipeline
├── config.py                 ← All settings & constants
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── hand_tracker.py       ← MediaPipe wrapper
│   ├── feature_extractor.py  ← Landmark → 88-D features
│   ├── gesture_model.py      ← Keras model + trainer + predictor
│   ├── dataset_manager.py    ← Data collection & loading
│   ├── tts_engine.py         ← Text-to-speech engine
│   └── utils.py              ← Smoothing, FPS, HUD rendering
│
├── data/
│   ├── raw/                  ← Per-class CSV files (one row = one sample)
│   └── processed/            ← Merged numpy arrays after preprocessing
│
├── models/                   ← Saved Keras model + label-encoder + scaler
├── logs/                     ← Training history, metrics, plots
└── assets/                   ← Screenshots
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
# Python 3.9–3.11 recommended
pip install -r requirements.txt
```

### 2. Collect gesture data

```bash
# Collect all 38 gesture classes (webcam required)
python collect_data.py --all --samples 200

# Or collect just one gesture
python collect_data.py --gesture A

# Or one category
python collect_data.py --category commands
```

> **During collection:**
> - A countdown window opens — position your hand clearly in frame
> - The system auto-captures when it detects a hand
> - Press **Q** to skip to the next gesture

### 3. Train the model

```bash
python train_model.py
```

Output:
- `models/gesture_model_best.h5`
- `models/label_encoder.pkl`
- `models/feature_scaler.pkl`
- `logs/training_curves.png`
- `logs/confusion_matrix.png`

### 4a. Run real-time recognition (OpenCV)

```bash
python run_recognition.py

# Options:
python run_recognition.py --camera 1 --width 1280 --height 720 --no-tts
```

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `C` | Clear sentence |
| `SPACE` | Add space |
| `T` | Toggle TTS |
| `S` | Screenshot |
| `BACKSPACE` | Delete last char |
| `R` | Reset smoother |

### 4b. Launch Streamlit web app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧠 Model Architecture

```
Input (88-D) ──────────────────────────────────────────────────────────────────┐
                                                                               │
Dense(512) → BatchNorm → ReLU → Dropout(0.4) ──→ Add ─────────────────────────│─┐
      └──────────── Skip(512) ─────────────────────┘                           │ │
                                                                               │ │
Dense(256) → BatchNorm → ReLU → Dropout(0.4) ──→ Add ─────────────────────────│─│─┐
      └──────────── Skip(256) ─────────────────────┘                           │ │ │
                                                                               │ │ │
Dense(128) → BatchNorm → ReLU → Dropout(0.4) ──→ Add ─────────────────────────│─│─│─┐
      └──────────── Skip(128) ─────────────────────┘                           │ │ │ │
                                                                               │ │ │ │
Dense(64)  → BatchNorm → ReLU → Dropout(0.4) ──→ Add                          │ │ │ │
      └──────────── Skip(64) ──────────────────────┘                           │ │ │ │
                                                                               │ │ │ │
Dense(NUM_CLASSES) → Softmax                                                   └─┘─┘─┘
```

**Training setup:**
- Optimiser: Adam (lr=0.001, ReduceLROnPlateau)
- Loss: Sparse Categorical Crossentropy
- EarlyStopping patience: 15 epochs
- Data split: 70% train / 20% val / 10% test
- Augmentation: feature noise + landmark rotation/scale

---

## 📐 Feature Vector (88 dimensions)

| # | Feature Group | Dim | Description |
|---|--------------|-----|-------------|
| 1 | Normalised coordinates | 63 | All 21 landmarks (x, y, z), translated to wrist origin, scaled by wrist→middle-MCP distance |
| 2 | Joint angles | 15 | Cosine angles at each finger joint triplet |
| 3 | Inter-landmark distances | 10 | Fingertip-to-fingertip Euclidean distances, normalised by hand span |

---

## 🎯 Supported Gestures

### Alphabet (A–Z)
Static ASL-inspired hand shapes for each letter.

### Numbers (0–9)
Finger counting / number signs.

### Commands
| Gesture | Display |
|---------|---------|
| `hello` | 👋 Hello |
| `yes` | ✅ Yes |
| `no` | ❌ No |
| `stop` | 🛑 Stop |
| `thumbs_up` | 👍 Thumbs Up |
| `thumbs_down` | 👎 Thumbs Down |
| `peace` | ✌️ Peace |
| `ok` | 👌 OK |
| `point` | ☝️ Point |
| `rock` | 🤘 Rock |
| `call_me` | 🤙 Call Me |
| `i_love_you` | 🤟 I Love You |

---

## ⚙️ Configuration

Edit `config.py` to tune any parameter:

```python
SAMPLES_PER_CLASS    = 200   # frames per gesture during collection
CONFIDENCE_THRESHOLD = 0.65  # min confidence to display prediction
SMOOTHING_WINDOW     = 7     # frames for majority-vote smoothing
TTS_COOLDOWN_SEC     = 2.0   # seconds between TTS announcements

MODEL_CFG = {
    "hidden_units"  : [512, 256, 128, 64],
    "dropout_rate"  : 0.4,
    "learning_rate" : 1e-3,
    "batch_size"    : 64,
    "epochs"        : 100,
    "patience"      : 15,
}
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not found | Try `--camera 1` or `--camera 2` |
| Low accuracy | Collect more samples, ensure good lighting, vary hand positions |
| TTS not working | `pip install pyttsx3` or use `--no-tts` |
| MediaPipe install fails | Try `pip install mediapipe-silicon` on Apple Silicon |
| CUDA out of memory | Add `--batch-size 32` or use tensorflow-cpu |
| Streamlit WebRTC issues | Use `run_recognition.py` for real-time instead |

---

## 📊 Expected Performance

With 200 samples/class + 3× augmentation (~150k total samples):

| Metric | Expected |
|--------|---------|
| Validation accuracy | 93–97% |
| Test accuracy | 90–95% |
| Inference latency | < 15 ms/frame |
| FPS (640×480) | 25–30 FPS |

---

## 📄 License

MIT License — free for educational and commercial use.

---

## 🙏 Acknowledgements

- [MediaPipe](https://google.github.io/mediapipe/) by Google
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
