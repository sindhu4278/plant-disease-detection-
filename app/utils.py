import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Helper directory resolving
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")

# Fallback classes matching generated ones
DEFAULT_CLASSES = [
    "Apple___healthy",
    "Apple___Black_rot",
    "Corn___Common_rust",
    "Tomato___Bacterial_spot",
    "Tomato___healthy"
]

def load_classes():
    """Loads class list from the saved file or uses fallback."""
    classes_file = os.path.join(OUTPUTS_DIR, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes
    return DEFAULT_CLASSES

def parse_class_name(cls_name):
    """
    Parses a PlantVillage style class name into crop, disease, and healthy status.
    Example: 'Tomato___Bacterial_spot' -> 'Tomato', 'Bacterial Spot', 'Diseased'
    """
    parts = cls_name.split("___")
    crop = parts[0].replace("_", " ").title()
    
    if len(parts) > 1:
        disease_raw = parts[1].replace("_", " ")
        if disease_raw.lower() == "healthy":
            disease = "None (Healthy)"
            status = "Healthy"
        else:
            disease = disease_raw.title()
            status = "Diseased"
    else:
        disease = "Unknown"
        status = "Unknown"
        
    return crop, disease, status

def get_treatment_recommendation(disease_name):
    """Returns treatment recommendations for a given disease name."""
    recommendations = {
        "Black Rot": (
            "1. Remove and destroy infected leaves, mummified fruits, and prune cankers from branches.\n"
            "2. Ensure good air circulation through proper pruning.\n"
            "3. Apply organic copper-based fungicides during the early spring bud break.\n"
            "4. Keep the ground clean around the plant base."
        ),
        "Common Rust": (
            "1. Plant rust-resistant varieties in future cycles.\n"
            "2. Apply fungicides (e.g., strobilurins or triazoles) at the first sign of pustules.\n"
            "3. Destroy infected crop residue after harvesting to prevent overwintering spores.\n"
            "4. Optimize nitrogen fertilization."
        ),
        "Bacterial Spot": (
            "1. Avoid overhead watering to limit leaf wetness duration.\n"
            "2. Apply copper-based bactericides combined with mancozeb weekly during wet spells.\n"
            "3. Prune low-hanging branches and remove infected debris immediately.\n"
            "4. Practice a 2-3 year crop rotation with non-solanaceous crops."
        ),
        "None (Healthy)": (
            "No disease detected! Continue maintaining standard cultural practices:\n"
            "1. Water at the base of the plant to avoid wet foliage.\n"
            "2. Ensure adequate spacing and ventilation.\n"
            "3. Monitor weekly for any visual shifts."
        )
    }
    return recommendations.get(disease_name, "Apply appropriate organic or chemical control agents according to local extension office guidance.")

def mock_predict(image):
    """
    Predicts a disease using a mock heuristic when models are not compiled.
    Looks at color profile of the image to make it respond realistically.
    """
    img_np = np.array(image.convert("RGB").resize((32, 32)))
    # Calculate green vs red/yellow ratio
    r_mean = np.mean(img_np[:, :, 0])
    g_mean = np.mean(img_np[:, :, 1])
    b_mean = np.mean(img_np[:, :, 2])
    
    classes = load_classes()
    
    # Fallback simulation
    # If image is predominantly reddish/yellowish brown, classify as disease
    if r_mean > g_mean * 0.95:
        # Predict corn rust or apple rot or tomato spot randomly based on red value
        if r_mean > 160:
            pred_idx = classes.index("Corn___Common_rust") if "Corn___Common_rust" in classes else 2
        elif g_mean > 130:
            pred_idx = classes.index("Tomato___Bacterial_spot") if "Tomato___Bacterial_spot" in classes else 3
        else:
            pred_idx = classes.index("Apple___Black_rot") if "Apple___Black_rot" in classes else 1
    else:
        # Predict healthy
        if g_mean > 120 and r_mean < 110:
            pred_idx = classes.index("Tomato___healthy") if "Tomato___healthy" in classes else 4
        else:
            pred_idx = classes.index("Apple___healthy") if "Apple___healthy" in classes else 0
            
    # Normalize index to list boundary
    pred_idx = min(max(0, pred_idx), len(classes) - 1)
    
    # Generate realistic pseudo-confidence (e.g. 88% - 98%)
    confidence = 0.85 + (float(r_mean + g_mean) % 15) / 100.0
    
    # Create fake probabilities
    probs = np.zeros(len(classes))
    probs[pred_idx] = confidence
    remaining = 1.0 - confidence
    for i in range(len(classes)):
        if i != pred_idx:
            probs[i] = remaining / (len(classes) - 1)
            
    return classes[pred_idx], confidence, probs

def load_keras_model(model_name):
    """Loads a compiled Keras model from disk using streamlit cache."""
    model_filename = f"{model_name.lower().replace(' ', '_')}.h5"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    if not os.path.exists(model_path):
        return None
        
    try:
        # Load model using keras
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def predict_image(image, model_name):
    """
    Runs model inference on a PIL image.
    If the model cannot be loaded, falls back to mock_predict with a warning.
    """
    classes = load_classes()
    model = load_keras_model(model_name)
    
    if model is None:
        # Model file not found, use mock prediction for fallback preview
        pred_class, confidence, probs = mock_predict(image)
        return pred_class, confidence, probs, True
        
    try:
        # Preprocess PIL image (convert to RGB to handle alpha channels and grayscale)
        img = image.convert("RGB").resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        
        # Predict
        predictions = model.predict(img_array)
        pred_idx = np.argmax(predictions[0])
        pred_class = classes[pred_idx]
        confidence = float(predictions[0][pred_idx])
        
        return pred_class, confidence, predictions[0], False
    except Exception as e:
        print(f"Prediction failed with loaded model, falling back to mock: {e}")
        pred_class, confidence, probs = mock_predict(image)
        return pred_class, confidence, probs, True
