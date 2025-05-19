import os
import json
import numpy as np
import pandas as pd
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from django.conf import settings
from PIL import Image
import base64
from io import BytesIO
from django.core.files.uploadedfile import InMemoryUploadedFile


# Load the preprocessed data and class indices
train_df = pd.read_csv(os.path.join(settings.BASE_DIR, 'dataset/train_data.csv'))
val_df = pd.read_csv(os.path.join(settings.BASE_DIR, 'dataset/val_data.csv'))
test_df = pd.read_csv(os.path.join(settings.BASE_DIR, 'dataset/test_data.csv'))

with open(os.path.join(settings.BASE_DIR, 'dataset/class_indices.json'), 'r') as f:
    class_indices = json.load(f)

# Reverse the class indices to get a mapping from index to class
index_to_class = {v: k for k, v in class_indices.items()}


# Load the pre-trained model
model = load_model(os.path.join(settings.BASE_DIR, 'EfficientNetB3_model.h5'))

temp_dir = os.path.join(settings.MEDIA_ROOT, 'tmp')
os.makedirs(temp_dir, exist_ok=True)

def index(request):
    return render(request, 'classifier/index.html')

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Ensure image size matches model input
    img = img_to_array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Handle file uploads and predictions
def start_diagnosing(request):
    if request.method == 'POST' and request.FILES['image']:
        file = request.FILES['image']
        file_path = os.path.join(temp_dir, file.name)
        
        # Save the uploaded file temporarily
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        
        # Encode the image to Base64 for display
        with open(file_path, 'rb') as image_file:
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Preprocess the image
        img = preprocess_image(file_path)

        # Make predictions
        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = index_to_class[predicted_class_index]
        predicted_probability = np.max(predictions)

        # Get disease description
        disease_description = disease_info.get(predicted_class_name, "Description not available.")

        # Clean up and delete the temporary image file
        if os.path.exists(file_path):
            os.remove(file_path)

        # Return the prediction and probability to the template
        return render(request, 'classifier/StartDiagnosing.html', {
            'predicted_class_name': predicted_class_name,
            'predicted_probability': predicted_probability,
            'disease_description': disease_description,
            'image_base64': image_base64
        })
    
    return render(request, 'classifier/StartDiagnosing.html')
    

def load_disease_info():
    file_path = os.path.join(settings.BASE_DIR, 'dataset/disease_info.json')
    with open(file_path, 'r') as f:
        return json.load(f)
    
# Load disease information
disease_info = load_disease_info()
