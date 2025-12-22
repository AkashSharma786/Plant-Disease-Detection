from flask import Flask, request, jsonify
import os
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variable - lazy loaded
model = None
tf = None

def load_model():
    """Load model on first use (lazy loading)"""
    global model, tf
    if model is not None:
        return
    
    logger.info("Loading PyTorch model...")
    model = models.mobilenet_v2(pretrained=True)

    for p in model.features.parameters():
        p.requires_grad = False

    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.classifier[1].in_features, 38)
    )
    model.load_state_dict(torch.load('mobilenetv2_plant.pth', map_location=torch.device('cpu')))
    model.eval()

    # Image preprocessing
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    logger.info("Model loaded successfully")

CLASS_NAMES = ["Apple___Apple_scab",
               "Apple___Black_rot",
               "Apple___Cedar_apple_rust",
               "Apple___healthy",
               "Blueberry___healthy",
               "Cherry_(including_sour)___Powdery_mildew",
               "Cherry_(including_sour)___healthy",
               "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
               "Corn_(maize)__Common_rust",
               "Corn_(maize)___Northern_Leaf_Blight",
               "Corn_(maize)___healthy",
               "Grape___Black_rot",
               "Grape___Esca_(Black_Measles)",
               "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
               "Grape___healthy",
               "Orange___Haunglongbing_(Citrus_greening)",
               "Peach___Bacterial_spot",
               "Peach___healthy",
               "Pepper,_bell___Bacterial_spot",
               "Pepper,_bell___healthy",
               "Potato___Early_blight",
               "Potato___Late_blight",
               "Potato___healthy",
               "Raspberry___healthy",
               "Soybean___healthy",
               "Squash___Powdery_mildew",
               "Strawberry___Leaf_scorch",
               "Strawberry___healthy",
               "Tomato___Bacterial_spot",
               "Tomato___Early_blight",
               "Tomato___Late_blight",
               "Tomato___Leaf_Mold",
               "Tomato___Septoria_leaf_spot",
               "Tomato___Spider_mites Two-spotted_spider_mite",
               "Tomato___Target_Spot",
               "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
               "Tomato___Tomato_mosaic_virus",
               "Tomato___healthy"]

DISEASE_CURES = {
    "Apple___Apple_scab": "Remove infected leaves, apply sulfur or copper fungicide sprays, improve air circulation, prune affected branches",
    "Apple___Black_rot": "Remove cankers and dead wood, apply protective fungicides, improve drainage, destroy infected fruit",
    "Apple___Cedar_apple_rust": "Remove galls from infected cedar trees, apply fungicide in spring, avoid cedar trees near apple orchards",
    "Apple___healthy": "No treatment needed - plant is healthy",
    "Blueberry___healthy": "No treatment needed - plant is healthy",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur or potassium bicarbonate spray, improve air circulation, remove infected leaves, avoid overhead watering",
    "Cherry_(including_sour)___healthy": "No treatment needed - plant is healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Remove lower leaves, apply fungicide, crop rotation, improve air circulation, remove debris",
    "Corn_(maize)__Common_rust": "Apply fungicide, remove infected leaves, improve air circulation, use resistant varieties, crop rotation",
    "Corn_(maize)___Northern_Leaf_Blight": "Apply fungicide containing azoxystrobin, use resistant hybrids, remove lower leaves, crop rotation",
    "Corn_(maize)___healthy": "No treatment needed - plant is healthy",
    "Grape___Black_rot": "Remove infected fruit and canes, apply fungicide, improve air circulation, remove debris, prune crowded canes",
    "Grape___Esca_(Black_Measles)": "Prune out infected canes below disease symptoms, apply pruning wound sealant, remove dead wood, improve drainage",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicide, remove infected leaves, improve air circulation, crop rotation, avoid overhead watering",
    "Grape___healthy": "No treatment needed - plant is healthy",
    "Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees, manage psyllid vectors with insecticides, use disease-free nursery stock, quarantine procedures",
    "Peach___Bacterial_spot": "Apply copper fungicide, prune infected branches, improve air circulation, avoid overhead watering, remove debris",
    "Peach___healthy": "No treatment needed - plant is healthy",
    "Pepper,_bell___Bacterial_spot": "Apply copper-based bactericide, remove infected leaves, improve air circulation, avoid overhead watering, crop rotation",
    "Pepper,_bell___healthy": "No treatment needed - plant is healthy",
    "Potato___Early_blight": "Apply fungicide, remove lower leaves, improve air circulation, crop rotation, avoid overhead watering, remove debris",
    "Potato___Late_blight": "Apply fungicide containing mancozeb or chlorothalonil, remove infected plants, improve air circulation, crop rotation",
    "Potato___healthy": "No treatment needed - plant is healthy",
    "Raspberry___healthy": "No treatment needed - plant is healthy",
    "Soybean___healthy": "No treatment needed - plant is healthy",
    "Squash___Powdery_mildew": "Apply sulfur or potassium bicarbonate spray, improve air circulation, remove infected leaves, avoid overhead watering",
    "Strawberry___Leaf_scorch": "Remove infected leaves, apply fungicide, improve air circulation, avoid overhead watering, crop rotation, remove runners",
    "Strawberry___healthy": "No treatment needed - plant is healthy",
    "Tomato___Bacterial_spot": "Apply copper-based bactericide, remove infected leaves, improve air circulation, avoid overhead watering, crop rotation",
    "Tomato___Early_blight": "Apply fungicide, remove lower leaves, improve air circulation, avoid overhead watering, crop rotation, mulch soil",
    "Tomato___Late_blight": "Apply fungicide, remove infected plants, improve air circulation, crop rotation, ensure proper drainage, prune dense foliage",
    "Tomato___Leaf_Mold": "Improve air circulation, apply fungicide, prune dense foliage, avoid overhead watering, lower humidity, crop rotation",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves, apply fungicide, improve air circulation, avoid overhead watering, crop rotation, remove debris",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Apply miticide or insecticidal soap, increase humidity, improve air circulation, remove heavily infested leaves, spray water",
    "Tomato___Target_Spot": "Remove infected leaves, apply fungicide, improve air circulation, avoid overhead watering, crop rotation, remove debris",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants, manage whitefly vectors with insecticides, use reflective mulch, plant resistant varieties",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants, disinfect tools and hands between plants, use resistant varieties, control aphid vectors",
    "Tomato___healthy": "No treatment needed - plant is healthy"
}


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts an image file and returns prediction with confidence
    """
    try:
        # Load model on first request (lazy loading)
        load_model()
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and process image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        x = tf(img).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            preds = model(x)
            probs = torch.softmax(preds, dim=1)[0]
            index = probs.argmax().item()
            confidence = float(probs[index])
        
        predicted_class = CLASS_NAMES[index]
        cure = DISEASE_CURES.get(predicted_class, "Treatment information not available")
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'cure': cure,
            'all_predictions': {
                CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
            }
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'Server is running'}), 200


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
