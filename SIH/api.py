from fastapi import FastAPI
import numpy as np
import tensorflow as tf
from pydantic import BaseModel
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('crop_recommendation_model.keras')

# Load the scalers
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the label encoders
with open('label_encoder_N.pkl', 'rb') as f:
    label_encoder_N = pickle.load(f)
with open('label_encoder_P.pkl', 'rb') as f:
    label_encoder_P = pickle.load(f)
with open('label_encoder_K.pkl', 'rb') as f:
    label_encoder_K = pickle.load(f)
with open('label_encoder_labels.pkl', 'rb') as f:
    label_encoder_labels = pickle.load(f)

# Crop labels
crop_labels = label_encoder_labels.classes_

# Input data structure
class CropInput(BaseModel):
    Nitrogen: str
    Phosphorus: str
    Potassium: str
    Temperature: float
    Humidity: float
    Ph: float
    Rainfall: float

# Function to preprocess input
def preprocess_input(input_data: CropInput):
    nitrogen_encoded = label_encoder_N.transform([input_data.Nitrogen])[0]
    phosphorus_encoded = label_encoder_P.transform([input_data.Phosphorus])[0]
    potassium_encoded = label_encoder_K.transform([input_data.Potassium])[0]
    
    # Combine the encoded features and numerical values
    features = np.array([[nitrogen_encoded, phosphorus_encoded, potassium_encoded,
                          input_data.Temperature, input_data.Humidity, input_data.Ph, input_data.Rainfall]])
    
    # Apply scaling
    scaled_features = scaler.transform(features)
    return scaled_features

# Prediction endpoint
@app.post("/predict/")
async def predict_crop(input_data: CropInput):
    # Preprocess input
    processed_input = preprocess_input(input_data)
    
    # Make a prediction
    prediction = model.predict(processed_input)
    
    # Convert the prediction to the crop label
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = crop_labels[predicted_class[0]]
    
    # Get the confidence score
    confidence_score = prediction[0][predicted_class[0]]
    
    return {
        "predicted_crop": predicted_label,
        "confidence": float(confidence_score)  # Convert to float for JSON serialization
    }
