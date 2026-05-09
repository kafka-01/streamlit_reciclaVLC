import streamlit as st
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

import re

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
        "Resto": "icon-gris",
        "Vidrio": "icon-verde",
        "Pilas": "icon-pilas",
        "Aceite usado": "icon-aceite",
        "Ecoparque móvil": "icon-rojo",
        "Ropa": "icon-ropa",
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
labels = ['Papel / Carton', 'Vidrio', 'Ecoparque móvil', 'Papel / Carton', 'Envases Ligeros', 'Resto']
    
# Secret
my_email = st.secrets['email']
model_weight_file = st.secrets['model_url']

import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom")
class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        # Fuerza a lista
        if not isinstance(inputs, (list, tuple)):
            return inputs * self.scale

        # Escala cada rama
        scaled = [tf.cast(x, tf.float32) * self.scale for x in inputs]

        # Fusión: UN SOLO tensor
        return tf.add_n(scaled)

    def compute_output_shape(self, input_shape):
        # Devuelve la shape de una sola rama
        if isinstance(input_shape, (list, tuple)):
            return input_shape[0]
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"scale": self.scale})
        return cfg
    
# Load the model into cache at the beginning of execution
@st.cache_resource(show_spinner = False)
def cargar_modelo():    
    if not os.path.exists('./models/model.h5'):
        u = urlopen(model_weight_file)
        data = u.read()
        u.close()
        with open(model_path, 'wb') as f:
            f.write(data)
    
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "CustomScaleLayer": CustomScaleLayer,
            "Custom>CustomScaleLayer": CustomScaleLayer
        }
    )

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
    image = image.resize(scaled_size, 3)

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
    
    return predicted_class, probability

def get_probability_text(probability):
    if probability < 0.3:
        return "Hmm... parece que la clasificación no está del todo clara. Puede que necesitemos más datos o un ajuste en el modelo. Sigamos trabajando para mejorar."
    elif probability < 0.6:
        return "La clasificación es relativamente segura, aunque aún existe un margen de error. Sigamos refinando nuestros conocimientos en el reciclaje para ofrecer resultados más precisos."
    elif probability < 0.9:
        return "¡Excelente! La clasificación es bastante segura, lo cual es alentador. Nuestra dedicación al reciclaje y la conciencia ecológica están dando frutos."
    else:
        return "¡Increíble! La clasificación es muy segura. Nuestro compromiso con el reciclaje y la protección del medio ambiente está dando resultados notables. Sigamos cuidando de nuestro planeta."

def identify_waste_app():
    
    uploaded_file = st.file_uploader('Sube una imagen con fondo blanco del objeto a identificar', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        image = preprocess_image(image)

        with st.spinner("Loading model for first time..."):
        
            model = cargar_modelo()
            
        with st.spinner("Predicting..."):
        
            predicted_class, prob = predict(image, model)
        
            # Get the icon and safety text
            icon = get_icon(predicted_class)
            prob_text = get_probability_text(prob)

            # Display the image and classification
            col1, col2 = st.columns([2, 4])  # Divide the space into two columns
            with col1:
                st.image(image, caption=f'{predicted_class} ({prob:.2f})')
                #st.image(image)

            # Display the icon and safety text
            with col2:
                st.image(Image.open(icon), width=50)  # Display the icon as an image
                st.markdown(f"**{predicted_class}**\n{prob_text}")  # Display the safety text with Markdown

# CONTAINER LOCATION =================================================================================

# PAGE ======================================================================================================    

# Logo
st.image(img_reciclaVLC)

# Title
st.title("Recicla Valencia: Cuida tu ciudad, separa tu basura")

# Initial block
intro_text = """
## ¡Bienvenido a Recicla Valencia!
Recicla Valencia es una plataforma dedicada a promover el reciclaje y la separación adecuada de la basura en nuestra hermosa ciudad. Utilizamos la ciencia de datos para concienciar a los vecinos de Valencia sobre la importancia de estas prácticas para preservar el medio ambiente y construir un futuro sostenible.

Nuestro objetivo es utilizar la tecnología y los datos para informar y motivar a los ciudadanos a reciclar de manera adecuada. Creemos firmemente que cada pequeño gesto cuenta y que juntos podemos marcar la diferencia en el cuidado de nuestro entorno.

## ¿Por qué reciclar?
Reciclar es fundamental para combatir los problemas asociados con la acumulación de basura y la falta de reciclaje. La contaminación del suelo, agua y aire, el agotamiento de recursos naturales no renovables y la generación de gases de efecto invernadero son solo algunas de las consecuencias negativas de no tomar acción. Estadísticas alarmantes, como la cantidad de residuos generados y la baja tasa de reciclaje actual, respaldan la necesidad de actuar de manera consciente y responsable.

## ¿Cómo empezar?
Comenzar a reciclar en casa es más fácil de lo que piensas. Te guiamos paso a paso para la separación de la basura:

1. Descubre la red de recogida selectiva y contenedores específicos en tu barrio.
2. Identifica los diferentes tipos de residuos y aprende cómo prepararlos para su correcta disposición.
3. Te brindamos consejos prácticos para reducir la generación de residuos y reutilizar objetos en el hogar, fomentando así una mentalidad más sostenible.

## Reciclaje en la comunidad
En Valencia, contamos con una amplia red de puntos de recogida selectiva y contenedores específicos para cada tipo de residuo. Te proporcionamos información detallada sobre los diferentes contenedores y qué residuos se deben depositar en cada uno. Además, te orientamos sobre los programas y servicios municipales relacionados con el reciclaje, como la recogida puerta a puerta y la gestión de residuos peligrosos.

### Recogida selectiva y contenedores

En Valencia, el sistema de separación de la basura se organiza en varios contenedores de diferentes colores. A continuación, te proporcionamos una guía básica sobre cómo separar la basura en los contenedores de Valencia:

- Contenedor Amarillo: Residuos de envases y embalajes de plástico, latas y briks. Aquí debes depositar elementos como botellas de plástico, latas de refresco, envases de yogur, bolsas de plástico, cartones de leche, entre otros.
- Contenedor Azul: Residuos de papel y cartón. Puedes depositar periódicos, revistas, folletos, cajas de cartón, papel de envolver, etc. Es importante que pliegues o desmontes las cajas para aprovechar mejor el espacio.
- Contenedor Verde: Residuos de vidrio. Puedes separar botellas de vidrio, frascos, tarros y otros envases de vidrio. No introduzcas tapas de metal o plástico, ya que deben ser depositadas en el contenedor correspondiente.
- Contenedor Marrón: Residuos orgánicos o restos de comida. Puedes depositar cáscaras de frutas y verduras, restos de comida, posos de café, restos de poda, entre otros materiales orgánicos. No introduzcas plásticos ni otros materiales no biodegradables.
- Contenedor Gris: Residuos no reciclables. Aquí debes depositar elementos como pañales, compresas, papel higiénico, colillas, chicles, etc. Trata de reducir al mínimo la cantidad de residuos que se depositan en este contenedor.

### Otros contenedores
En nuestra comunidad, contamos con contenedores especiales para la recolección de aceite usado y pilas. El aceite de cocina usado debe ser depositado en botellas de plástico cerradas en el contenedor correspondiente, donde será reciclado para producir biodiesel. Las pilas, que contienen sustancias tóxicas, deben ser colocadas en los contenedores específicos para su correcta gestión y reciclaje, evitando la contaminación del suelo y del agua. Estos contenedores especiales son una parte importante de nuestro sistema de reciclaje y nos ayudan a proteger el medio ambiente. Participa en su uso adecuado para mantener nuestra comunidad limpia y sostenible.

### Ecoparques móviles
Además de los contenedores mencionados, en Valencia también existen los ecoparques o puntos limpios, donde se pueden depositar residuos especiales como pilas, baterías, electrodomésticos, muebles, aceites usados, etc. Estos residuos deben ser llevados personalmente a los ecoparques para su correcto tratamiento y reciclaje.
"""
st.markdown(intro_text)
    
# Identify waste
identify_text = '''
## Identificación de los residuos

Aprende a reconocer y clasificar los distintos tipos de residuos comunes, como orgánicos, papel y cartón, vidrio, plástico, metal, etc. Esto te ayudará a saber en qué contenedor depositar cada tipo de residuo.

Si tienes alguna duda sobre cómo reciclar un objeto específico, no dudes en utilizar nuestra herramienta de Identificación de Residuos. Sube una foto con fondo blanco del objeto en cuestión, y te ayudaremos a determinar en qué contenedor debe ser depositado.<sup><a name="back" href="#disclaimer">[1]</a></sup>
'''
st.markdown(identify_text, unsafe_allow_html=True)

# Identify waste
with st.container():    
    
    supported_classes = set(labels)
    # Crear una lista de anchos de columna
    column_widths = [64] + [50] * 10
    columns = st.columns(column_widths)    
    columns[0].image(img_reciclaVLC_app2, width=64)
    for i, c in enumerate(supported_classes):
        columns[i+1].image(Image.open(get_icon(c)), width=50)
    
    identify_waste_app()

# Final block
preparation_text = """
## Preparar los residuos
Es importante preparar los residuos antes de desecharlos para su correcta disposición. Algunas recomendaciones generales incluyen:
- Lavar los envases de plástico, vidrio y metal antes de desecharlos.
- Retirar tapas y etiquetas de los envases antes de reciclarlos.
- Aplastar o plegar las cajas y envases para ahorrar espacio en los contenedores.
## Reducción de residuos
Además de reciclar, es fundamental reducir la generación de residuos en primer lugar. Te proporcionamos consejos prácticos para adoptar hábitos más sostenibles en tu hogar, como:
- Utilizar bolsas reutilizables en lugar de bolsas de plástico.
- Comprar productos a granel para evitar envases innecesarios.
- Optar por productos duraderos y de calidad en lugar de artículos desechables.
- Reutilizar objetos y darles una segunda vida antes de desecharlos.

Recuerda que cada pequeño gesto cuenta, y al incorporar estas prácticas en tu rutina diaria, estarás contribuyendo significativamente al cuidado del medio ambiente y la construcción de un futuro más sostenible. ¡Empieza a reciclar y reducir tus residuos hoy mismo!

## Recursos adicionales
Accede a recursos educativos, como guías de reciclaje y tutoriales sobre reutilización creativa, a través de los enlaces que proporcionamos. Mantente informado sobre eventos y campañas relacionadas con el reciclaje en Valencia y descubre cómo colaborar con organizaciones locales comprometidas con el cuidado del medio ambiente.

- OpenData Valencia: [OpenData Valencia](https://valencia.opendatasoft.com)
- Ayuntamiento de Valencia: [València Neta](https://www.valencia.es/cas/vlcneta/inicio)
- Universidad Politécnica de Valencia: [Plástico cero en la UPV](http://medioambiente.webs.upv.es/plasticocero/)

¡Únete a nosotros en Recicla Valencia y juntos hagamos del reciclaje una prioridad en nuestra ciudad!
"""
st.markdown(preparation_text)

#Disclaimer
disclaimer_text = '<a name="disclaimer" href="#back"><sup>[1]</sup></a><small><b>Aviso</b>: Esta aplicación tiene fines educativos y proporciona información básica sobre el reciclaje. En caso de duda, se recomienda consultar las fuentes mencionadas en los recursos adicionales para obtener información más detallada y precisa.</small>'

st.markdown(disclaimer_text, unsafe_allow_html=True)
