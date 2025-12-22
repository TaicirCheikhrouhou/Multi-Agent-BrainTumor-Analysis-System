# tools/classifier_tool.py
from crewai.tools import tool
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_PATH = "models/best_model_VGG19.keras"
classifier_model = None

if os.path.exists(MODEL_PATH):
    try:
        print(f"[INFO] Loading VGG19 model from {MODEL_PATH}...")
        classifier_model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading error: {e}")
else:
    print(f"MISSING MODEL: {MODEL_PATH}")

def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess MRI image for model prediction.

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image array ready for prediction
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@tool("classify_brain_mri")
def classify_brain_mri(image_path: str) -> str:
    """
    Classify a brain MRI image to detect the presence of a tumor using a trained VGG19 model.

    Args:
        image_path (str): Path to the MRI image (PNG, JPG, or JPEG)

    Returns:
        str: Formatted result with diagnosis, confidence, and tumor probability
    """
    if not os.path.exists(image_path):
        return f"ERROR: Image not found → {image_path}"

    if classifier_model is None:
        return "ERROR: VGG19 model not loaded. Place 'best_model_VGG19.keras' in the 'models/' folder"

    try:
        img_array = preprocess_image(image_path)
        prediction = float(classifier_model.predict(img_array, verbose=0)[0][0])
        has_tumor = prediction > 0.5
        diagnosis = "Tumor detected" if has_tumor else "No tumor detected"
        confidence = prediction * 100 if has_tumor else (1 - prediction) * 100

        return f"""Diagnosis: {diagnosis}
Confidence: {confidence:.1f}%
Tumor probability: {prediction * 100:.1f}%"""

    except Exception as e:
        return f"Prediction error: {str(e)}"