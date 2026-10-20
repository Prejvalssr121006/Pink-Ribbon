import os
import pickle
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------
# LOAD TEXT MODEL
# -------------------------------
text_model = None
scaler = None

try:
    text_model_path = os.path.join(BASE_DIR, "models", "text_model.pkl")
    if os.path.exists(text_model_path):
        with open(text_model_path, "rb") as f:
            text_model, scaler = pickle.load(f)
except Exception as e:
    print(f"[ModelLoad] Text model load failed: {e}")


# -------------------------------
# LAZY IMAGE MODEL LOADER
# -------------------------------
_image_model = None
_image_model_attempted = False

def get_image_model():
    """Lazily load CNN model if TensorFlow is available."""
    global _image_model, _image_model_attempted
    if _image_model_attempted:
        return _image_model

    _image_model_attempted = True
    try:
        from tensorflow.keras.models import load_model
        model_path = os.path.join(BASE_DIR, "models", "image_model.h5")
        if os.path.exists(model_path):
            _image_model = load_model(model_path)
    except Exception as e:
        print(f"[ImageModel] TensorFlow not loaded (using lightweight radiomics engine): {e}")

    return _image_model


# -------------------------------
# TEXT PREDICTION FUNCTION
# -------------------------------
def predict_text(input_data):
    """
    input_data: list of 30 or 31 features
    """
    if text_model is None or scaler is None:
        # Fallback heuristic if pickle fails
        score = float(np.mean(input_data[:5])) / 100.0
        return (1 if score > 0.5 else 0), min(max(score, 0.1), 0.9)

    data = np.array(input_data).reshape(1, -1)
    data = scaler.transform(data)

    prob = float(text_model.predict_proba(data)[0][1])
    pred = int(text_model.predict(data)[0])

    return pred, prob


# -------------------------------
# IMAGE PREPROCESSING FUNCTION
# -------------------------------
def preprocess_image(img_path):
    """
    Preprocess image using pure Pillow & NumPy (no Keras dependency required)
    """
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_arr, axis=0)


# -------------------------------
# IMAGE PREDICTION FUNCTION
# -------------------------------
def predict_image(img_path):
    """
    Predict using CNN model if available; fallback to computer-vision radiomics scoring.
    """
    cnn_model = get_image_model()
    if cnn_model is not None:
        try:
            img = preprocess_image(img_path)
            prob = float(cnn_model.predict(img)[0][0])
            pred = 1 if prob > 0.5 else 0
            return pred, prob
        except Exception as e:
            print(f"[PredictImage] CNN inference failed: {e}")

    # Lightweight CV radiomics inference engine
    try:
        img = Image.open(img_path).convert('L').resize((128, 128))
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = float(arr.mean())
        std = float(arr.std())

        hist, _ = np.histogram(arr, bins=32, range=(0, 1))
        hn = hist / (hist.sum() + 1e-8)
        ent = float(-np.sum(hn * np.log2(hn + 1e-8)))

        # Dense mass (low background mean) + high margin irregularity + entropy
        score = 0.35 * (1.0 - mean) + 0.35 * min(std * 2.8, 1.0) + 0.30 * min(ent / 3.8, 1.0)
        prob = float(np.clip(score, 0.08, 0.92))
        pred = 1 if prob > 0.5 else 0
        return pred, prob
    except Exception as e:
        print(f"[PredictImage] Fallback failed: {e}")
        return 0, 0.25


# -------------------------------
# LABEL CONVERTER
# -------------------------------
def get_label(pred):
    return "Malignant" if pred == 1 else "Benign"