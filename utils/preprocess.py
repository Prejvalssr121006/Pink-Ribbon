import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# -------------------------------
# LOAD MODELS (LOAD ONLY ONCE)
# -------------------------------

# Text model
with open("models/text_model.pkl", "rb") as f:
    text_model, scaler = pickle.load(f)

# Image model
image_model = load_model("models/image_model.h5")


# -------------------------------
# TEXT PREDICTION FUNCTION
# -------------------------------

def predict_text(input_data):
    """
    input_data: list of 30 features
    """

    data = np.array(input_data).reshape(1, -1)
    data = scaler.transform(data)

    prob = text_model.predict_proba(data)[0][1]
    pred = text_model.predict(data)[0]

    return pred, prob


# -------------------------------
# IMAGE PREPROCESSING FUNCTION
# -------------------------------

def preprocess_image(img_path):
    """
    Preprocess uploaded image
    """

    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)

    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)

    return img


# -------------------------------
# IMAGE PREDICTION FUNCTION
# -------------------------------

def predict_image(img_path):
    """
    Predict using CNN model
    """

    img = preprocess_image(img_path)

    prob = image_model.predict(img)[0][0]
    pred = 1 if prob > 0.5 else 0

    return pred, prob


# -------------------------------
# LABEL CONVERTER (OPTIONAL)
# -------------------------------

def get_label(pred):
    return "Malignant" if pred == 1 else "Benign"