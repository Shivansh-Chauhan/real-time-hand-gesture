"""
gesture_model.py
────────────────
Defines, trains, evaluates and loads the gesture-classification model.

Architecture
────────────
Dense MLP with residual skip-connections, batch-normalisation and dropout.
Input  : 88-dimensional feature vector (normalised landmarks + angles + distances)
Output : softmax over NUM_CLASSES gesture classes
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from tensorflow.keras import layers, Model, callbacks

from config import (
    MODEL_CFG, TOTAL_FEATURE_DIM, NUM_CLASSES, ALL_GESTURES,
    BEST_MODEL_PATH, LABEL_ENCODER_PATH, SCALER_PATH, HISTORY_PATH,
    MODEL_DIR, LOGS_DIR,
)


# ── model definition ─────────────────────────────────────────────────────────

def _dense_block(x, units: int, dropout: float, name_prefix: str):
    """Dense → BatchNorm → Activation → Dropout."""
    x = layers.Dense(units, use_bias=False, name=f"{name_prefix}_dense")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_act")(x)
    x = layers.Dropout(dropout, name=f"{name_prefix}_drop")(x)
    return x


def build_model(
    input_dim: int = TOTAL_FEATURE_DIM,
    num_classes: int = NUM_CLASSES,
    hidden_units: Optional[List[int]] = None,
    dropout_rate: float = 0.4,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Build and compile the MLP gesture classifier.

    The network uses residual-like projections when adjacent block sizes match,
    providing gradient highways during deep training.
    """
    if hidden_units is None:
        hidden_units = MODEL_CFG["hidden_units"]

    inp = layers.Input(shape=(input_dim,), name="landmarks_input")
    x   = inp

    for i, units in enumerate(hidden_units):
        prefix = f"block{i+1}"
        # Optional skip connection (project if shapes differ)
        if x.shape[-1] == units:
            skip = x
        else:
            skip = layers.Dense(units, use_bias=False, name=f"{prefix}_skip")(x)

        x = _dense_block(x, units, dropout_rate, prefix)
        x = layers.Add(name=f"{prefix}_add")([x, skip])

    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inp, out, name="GestureNet")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── trainer ──────────────────────────────────────────────────────────────────

class GestureModelTrainer:
    """Handles full training pipeline: preprocessing → fit → evaluate → save."""

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self.label_encoder = LabelEncoder()
        self.scaler        = StandardScaler()
        self.model: Optional[Model] = None
        self.history: Dict = {}

    # ── data prep ────────────────────────────────────────────────────────────

    def prepare_data(
        self,
        X: np.ndarray,
        y_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """
        Encode labels, scale features, and split into train/val/test.

        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        from sklearn.model_selection import train_test_split

        y_enc = self.label_encoder.fit_transform(y_labels)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        test_size = MODEL_CFG["test_split"]
        val_size  = MODEL_CFG["val_split"]

        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X_scaled, y_enc, test_size=test_size, random_state=42, stratify=y_enc
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp,
            test_size=val_size / (1 - test_size),
            random_state=42, stratify=y_tmp
        )
        print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ── training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        num_classes: int,
    ) -> tf.keras.callbacks.History:
        """Build and fit the model."""
        self.model = build_model(
            input_dim    = X_train.shape[1],
            num_classes  = num_classes,
            hidden_units = MODEL_CFG["hidden_units"],
            dropout_rate = MODEL_CFG["dropout_rate"],
            learning_rate= MODEL_CFG["learning_rate"],
        )
        self.model.summary()

        cb_list = [
            callbacks.EarlyStopping(
                monitor="val_accuracy", patience=MODEL_CFG["patience"],
                restore_best_weights=True, verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=7,
                min_lr=1e-6, verbose=1,
            ),
            callbacks.ModelCheckpoint(
                str(BEST_MODEL_PATH), monitor="val_accuracy",
                save_best_only=True, verbose=0,
            ),
        ]
        # TensorBoard is optional — skip if tensorboard package is missing
        try:
            tb_cb = callbacks.TensorBoard(log_dir=str(LOGS_DIR / "tb"), histogram_freq=1)
            # Probe by accessing a harmless attribute
            _ = tb_cb.log_dir
            cb_list.append(tb_cb)
        except Exception:
            pass

        history = self.model.fit(
            X_train, y_train,
            validation_data = (X_val, y_val),
            epochs          = MODEL_CFG["epochs"],
            batch_size      = MODEL_CFG["batch_size"],
            callbacks       = cb_list,
            verbose         = 1,
        )
        self.history = history.history
        return history

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Compute accuracy, per-class report and confusion matrix.

        Returns a metrics dict (also printed to stdout).
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred      = np.argmax(y_pred_prob, axis=1)

        acc = accuracy_score(y_test, y_pred)
        class_names = self.label_encoder.classes_.tolist()

        report = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        )
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{'='*60}")
        print(f"  Test Accuracy : {acc*100:.2f}%")
        print(f"{'='*60}")
        print(classification_report(y_test, y_pred, target_names=class_names))

        return {
            "accuracy"         : float(acc),
            "classification_report": report,
            "confusion_matrix" : cm.tolist(),
            "class_names"      : class_names,
        }

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save model, label-encoder, scaler and training history."""
        if self.model is None:
            raise RuntimeError("Nothing to save.")

        self.model.save(str(BEST_MODEL_PATH))

        with open(LABEL_ENCODER_PATH, "wb") as f:
            pickle.dump(self.label_encoder, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(HISTORY_PATH, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n✅  Model saved → {BEST_MODEL_PATH}")
        print(f"✅  Label encoder → {LABEL_ENCODER_PATH}")
        print(f"✅  Scaler → {SCALER_PATH}")

    def load(self) -> None:
        """Load saved model + preprocessors from disk."""
        self.model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
        with open(LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)


# ── inference wrapper ─────────────────────────────────────────────────────────

class GesturePredictor:
    """
    Lightweight inference wrapper. Loads saved artefacts once and provides
    fast single-sample prediction.
    """

    def __init__(self):
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {BEST_MODEL_PATH}. "
                "Please run train_model.py first."
            )
        self.model: Model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
        with open(LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder: LabelEncoder = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler: StandardScaler = pickle.load(f)

        print(f"✅  GesturePredictor loaded — {len(self.label_encoder.classes_)} classes")

    def predict(
        self,
        feature_vec: np.ndarray,
    ) -> Tuple[str, float, np.ndarray]:
        """
        Classify a single feature vector.

        Parameters
        ----------
        feature_vec : (TOTAL_FEATURE_DIM,) float32

        Returns
        -------
        label      : predicted gesture name
        confidence : probability of top-1 class
        probs      : full softmax probability array
        """
        x = self.scaler.transform(feature_vec.reshape(1, -1))
        probs = self.model.predict(x, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        label = self.label_encoder.inverse_transform([top_idx])[0]
        return label, float(probs[top_idx]), probs

    def top_k(
        self,
        feature_vec: np.ndarray,
        k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Return top-k (label, confidence) pairs."""
        _, _, probs = self.predict(feature_vec)
        top_indices = np.argsort(probs)[::-1][:k]
        classes = self.label_encoder.classes_
        return [(classes[i], float(probs[i])) for i in top_indices]
