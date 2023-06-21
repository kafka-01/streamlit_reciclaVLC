​import streamlit as st
import os
from PIL import Image
from urllib.request import urlopen

import requests
import math
import folium
from streamlit_folium import folium_static

import tensorflow as tf
from tensorflow.keras.models import load_model

import numpy as np

# Get the absolute path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))

# Open the image files
img_reciclaVLC = Image.open(os.path.join(current_path, 'images', 'ReciclaVLC.png'))
img_reciclaVLC_app1 = Image.open(os.path.join(current_path, 'images', 'ReciclaVLC-Localiza.png'))
img_reciclaVLC_app2 = Image.open(os.path.join(current_path, 'images', 'ReciclaVLC-Identifica.png'))

#Other resources
model_path = os.path.join(current_path, 'models', 'model.h5')

def get_icon(label):
    iconos_reciclaje = {
        "Envases Ligeros": "icon-amarillo",
        "Organico": "icon-marron",
        "Papel / Carton": "icon-azul",
        "Residuos Urbanos": "icon-gris",
        "Vidrio": "icon-verde",
        "Pilas": "icon-pilas",
        "Aceite usado": "icon-aceite",
        "Ecoparque móvil": "icon-rojo",
        "Todos": ""
    }
    
    if label in iconos_reciclaje:
        icon_path = os.path.join(current_path, 'images', iconos_reciclaje[label] + '.png')
        return icon_path
    else:
        return ""

# WASTE IDENTIFICATION ==================================================================================================

# Classification labels
#labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
labels = ['Papel / Carton', 'Vidrio', 'Ecoparque móvil', 'Papel / Carton', 'Envases Ligeros', 'Residuos Urbanos']
    
# Secret
my_email = st.secrets['email']
model_weight_file = st.secrets['model_url']    
    
# Load the model into cache at the beginning of execution
@st.cache_resource
def cargar_modelo():    
    if not os.path.exists('./models/model.h5'):
        u = urlopen(model_weight_file)
        data = u.read()
        u.close()
        with open(model_path, 'wb') as f:
            f.write(data)
    model = load_model(model_path)
    return model

# Prepare the image for display on the web and preprocess it for the model (preproc1)
def preprocess_image(image):
    # Rotate the image if necessary
    if hasattr(image, '_getexif') and image._getexif() is not None:
        exif = image._getexif()
        orientation = exif.get(0x0112)
        if orientation is not None:
            if orientation == 1:
                pass  # No rotation is required
            elif orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)

    # Check if the image is already in RGB format
    if image.mode != "RGB":
        # Convert the image to RGB format and replace it
        image = image.convert("RGB")

    # Scale the image so that its smaller side measures 224 pixels
    width, height = image.size
    if width < height:
        scaled_size = (224, int(224 * height / width))
    else:
        scaled_size = (int(224 * width / height), 224)
    image = image.resize(scaled_size, Image.ANTIALIAS)

    return image

def predict(image, model):
    # Resize the image to the required input size of the model
    image = image.resize((224, 224))

    # Preprocess the image
    preprocessed_image = tf.keras.applications.inception_resnet_v2.preprocess_input(np.array(image))

    # Expand dimensions to match the expected input shape of the model
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    # Make predictions using the model
    predictions = model.predict(preprocessed_image)

    # Return the predicted class and its probability
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = labels[predicted_class_index]
    probability = predictions[0][predicted_class_index]
    
