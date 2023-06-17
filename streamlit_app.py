import streamlit as st
import requests
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from PIL import Image
import os
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout
import numpy as np

from urllib.request import urlopen

my_email = st.secrets['email']
model_weight_file = st.secrets['model_url']

def get_barrios():
    url = "https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=barris-barrios&q=&rows=-1"
    response = requests.get(url)
    data = response.json()
    return data["records"]

def get_contenedores(filtro):
    coordinates = filtro['coordinates'][0]  # Obtener la lista de coordenadas del polígono
    
    # Formatear las coordenadas del polígono
    coordinates_str = '%2C+'.join([f'({coord[1]}%2C+{coord[0]})' for coord in coordinates])
    
    url = f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=contenidors-residus-solids-contenidores-residuos-solidos&q=&rows=-1&geofilter.polygon=" + coordinates_str
    response = requests.get(url)
    
    #st.markdown(url)
    
    data = response.json()
    return data["records"]

def calculate_center_zoom(shape_barrio):
    coords_barrio = shape_barrio["coordinates"][0]

    latitudes = [coord[1] for coord in coords_barrio]
    longitudes = [coord[0] for coord in coords_barrio]

    min_lat = min(latitudes)
    max_lat = max(latitudes)
    min_lon = min(longitudes)
    max_lon = max(longitudes)

    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    zoom = 15  # Nivel de zoom inicial

    lat_extent = max_lat - min_lat
    lon_extent = max_lon - min_lon

    if lat_extent != 0 and lon_extent != 0:
        max_zoom = 18  # Zoom máximo permitido (ajusta según tus necesidades)
        width_pixels = 800  # Ancho del mapa en píxeles (ajusta según tus necesidades)
        deg_per_pixel = (lon_extent / width_pixels)  # Grados de longitud por píxel
        zoom = math.floor(math.log((360 * math.cos(math.radians(center_lat))) / deg_per_pixel, 2))
        zoom = min(max_zoom, zoom)

    return (center_lat, center_lon), zoom - 2

#labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
labels = ['Papel / Carton', 'VIDRIO', 'Ecoparque', 'Papel / Carton', 'Envases Ligeros', 'Residuos Urbanos']

def prediccion(image, model):
    image = image.resize((224, 224))

    imagen_preprocesada = tf.keras.applications.inception_resnet_v2.preprocess_input(np.array(image))

    imagen_preprocesada = np.expand_dims(imagen_preprocesada, axis=0)

    predicciones = model.predict(imagen_preprocesada)

    indice_clase_predicha = np.argmax(predicciones[0])    
    clase_predicha = labels[indice_clase_predicha]
    prob = predicciones[0][indice_clase_predicha]
    
    
    return clase_predicha, prob

diccionario_reciclaje = {
    "Todos": "red",
    "Envases Ligeros": "orange",
    "Organico": "green",
    "Papel / Carton": "blue",
    "Residuos Urbanos": "gray",
    "VIDRIO": "green",
    "Pilas": "black",
    "Aceite usado": "light brown",
    "Ecoparque": "purple"
}

# Get the absolute path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))

# Open the image files
img_reciclaVLC = Image.open(os.path.join(current_path, 'images', 'ReciclaVLC.png'))
img_reciclaVLC_app1 = Image.open(os.path.join(current_path, 'images', 'ReciclaVLC-Localiza.png'))
img_reciclaVLC_app2 = Image.open(os.path.join(current_path, 'images', 'ReciclaVLC-Identifica.png'))

#Path of the model
model_path = os.path.join(current_path, 'models', 'model.h5')

if not os.path.exists('./models/model.h5'):
    u = urlopen(model_weight_file)
    data = u.read()
    u.close()
    with open(model_path, 'wb') as f:
        f.write(data)

# Variable de bandera para verificar si el modelo ya ha sido cargado
modelo_cargado = False

# Cargar el modelo en caché al principio de la ejecución
@st.cache_resource 
def cargar_modelo():
    
    model = load_model(model_path)
    
    return model

# Display the image using Streamlit's image function
st.image(img_reciclaVLC)

st.title("Recicla Valencia: Cuida tu ciudad, separa tu basura")


"""
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

### Localización de los contenedores cercanos
Utiliza nuestra herramienta interactiva para encontrar los contenedores más cercanos a tu barrio. Selecciona tu barrio en el menú desplegable y podrás visualizar en el mapa los contenedores específicos. ¡Recuerda seguir las indicaciones y depositar los residuos en el contenedor correcto!

"""

st.image(img_reciclaVLC_app1, width=64)

with st.container():

    # Lista de tipos de residuos
    tipos_residuos = list(diccionario_reciclaje.keys())

    # Widget selectbox para seleccionar el tipo de residuo
    residuo_seleccionado = st.selectbox("Selecciona el tipo de residuo", tipos_residuos)

    # Desplegable de los 88 barrios de Valencia
    # Lista de los 88 barrios de Valencia
    #barrios = ['Algirós', 'Benicalap', 'Benimaclet', 'Camins al Grau', 'Campanar', 'Ciutat Vella', 'El Pla del Real', 'Extramurs', 'Jesús', 'L\'Olivereta', 'La Saïdia', 'Patraix', 'Poblats Marítims', 'Quatre Carreres', 'Rascanya', 'Alboraia', 'Albuixech', 'Alfara del Patriarca', 'Almàssera', 'Bonrepòs i Mirambell', 'Burjassot', 'Emperador', 'Godella', 'La Pobla de Farnals', 'Mislata', 'Moncada', 'Nàquera', 'Rocafort', 'Tavernes Blanques', 'Vinalesa', 'Xirivella', 'Xàtiva', 'Albal', 'Alcàsser', 'Aldaia', 'Alfafar', 'Benetússer', 'Beniparrell', 'Catarroja', 'Massanassa', 'Picanya', 'Picassent', 'Sedaví', 'Silla', 'Torrent', 'El Puig de Santa Maria', 'Faura', 'Massamagrell', 'Meliana', 'Museros', 'Puzol', 'Rafelbunyol', 'Sagunt', 'Albalat dels Sorells', 'Alboraya', 'Albuixech', 'Alfara del Patriarca', 'Almàssera', 'Benifairó de les Valls', 'Bonrepòs i Mirambell', 'Burjassot', 'Foios', 'Godella', 'La Pobla de Farnals', 'Massalfassar', 'Massamagrell', 'Meliana', 'Moncada', 'Museros', 'Nàquera', 'Pobla de Vallbona (la)', 'Puçol', 'Rafelbunyol', 'Rocafort', 'Sagunt', 'Tavernes Blanques', 'Tavernes de la Valldigna', 'València', 'Vinalesa']

    barrios = []
    shapes = []

    barrios_json = get_barrios()

    # Ordenar los registros por el campo 'nombre'
    barrios_json = sorted(barrios_json, key=lambda x: x['fields']['nombre'])

    for barrio in barrios_json:
        nombre = barrio["fields"]["nombre"]
        shape = barrio["fields"]["geo_shape"]
        barrios.append(' '.join(word.capitalize() for word in nombre.split()))
        shapes.append(shape)

    barrio_seleccionado = st.selectbox('Selecciona tu barrio', barrios)

    # Obtener la forma del barrio seleccionado
    indice_barrio = barrios.index(barrio_seleccionado)
    shape_barrio = shapes[indice_barrio]

    # Mostrar mapa de Valencia

    # Calcular la ubicación central y el nivel de zoom óptimos
    center, zoom = calculate_center_zoom(shape_barrio)

    # Crear el mapa centrado en el barrio con el nivel de zoom óptimo
    valencia_map = folium.Map(location=center, zoom_start=zoom)

    # Añadir un marcador en el barrio seleccionado
    #folium.Marker(location=[39.46975, -0.37739], popup="Valencia").add_to(valencia_map)

    # Añadir la forma del barrio seleccionado
    #folium.GeoJson(shape_barrio).add_to(valencia_map)

    # Mostrar los contenedores

    contenedores = {}

    # Obtener contenedores por tipo en el barrio seleccionado
    contenedores_json = get_contenedores(shape_barrio)
    for c in contenedores_json:
        tipo = c["fields"]["tipo_resid"]
        coords = c["fields"]["geo_shape"]

        # Verificar si la clave existe en el diccionario
        if tipo not in contenedores:
            contenedores[tipo] = []

        # Agregar las coordenadas a la lista correspondiente
        contenedores[tipo].append(coords)

    # Recorrer cada clave y agregar marcadores
    for tipo, coordenadas in contenedores.items():

        if (tipo==residuo_seleccionado or residuo_seleccionado == 'Todos'):
            color = diccionario_reciclaje.get(tipo)  # Obtener el color asociado al tipo de residuo del diccionario

            for coords in coordenadas:
                # Obtener las coordenadas del punto
                lat = coords["coordinates"][1]
                lon = coords["coordinates"][0]

                # Crear un marcador con un icono compatible con Folium (por ejemplo, "leaf") y el color correspondiente
                icon = folium.Icon(icon="leaf", color=color)
                marker = folium.Marker(location=[lat, lon], popup=tipo, icon=icon)

                # Agregar el marcador al mapa
                valencia_map.add_child(marker)

    # Mostrar el mapa en Streamlit
    folium_static(valencia_map)

st.markdown('''
## Identificar los residuos

Aprende a reconocer y clasificar los distintos tipos de residuos comunes, como orgánicos, papel y cartón, vidrio, plástico, metal, etc. Esto te ayudará a saber en qué contenedor depositar cada tipo de residuo.

Si tienes alguna duda sobre cómo reciclar un objeto específico, no dudes en utilizar nuestra herramienta de Identificación de Residuos. Sube una foto con fondo blanco del objeto en cuestión, y te ayudaremos a determinar en qué contenedor debe ser depositado.<sup><a name="back" href="#disclaimer">[1]</a></sup>
''', unsafe_allow_html=True)

# Formulario para subir una imagen
st.subheader('Identificación de los residuos')
st.image(img_reciclaVLC_app2, width=64)
with st.container():
    # Llamar a la función para cargar el modelo
    if not modelo_cargado:
        modelo = cargar_modelo()
        modelo_cargado = True
    uploaded_file = st.file_uploader('Sube una imagen con fondo blanco del objeto a identificar', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        #st.image(image, caption="Uploaded Image")
        clase_predicha, prob = prediccion(image, modelo)
        st.image(image, width = 250, caption=f'Clase predicha: {clase_predicha}, Probabilidad: {prob:.2f}')

# Preparación de residuos
"""
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
"""
# Recursos adicionales
"""
## Recursos adicionales
Accede a recursos educativos, como guías de reciclaje y tutoriales sobre reutilización creativa, a través de los enlaces que proporcionamos. Mantente informado sobre eventos y campañas relacionadas con el reciclaje en Valencia y descubre cómo colaborar con organizaciones locales comprometidas con el cuidado del medio ambiente.

- OpenData Valencia: [OpenData Valencia](https://valencia.opendatasoft.com)
- Ayuntamiento de Valencia: [València Neta](https://www.valencia.es/cas/vlcneta/inicio)
- Universidad Politécnica de Valencia: [Plástico cero en la UPV](http://medioambiente.webs.upv.es/plasticocero/)

¡Únete a nosotros en Recicla Valencia y juntos hagamos del reciclaje una prioridad en nuestra ciudad!
"""

st.markdown('<a name="disclaimer" href="#back"><sup>[1]</sup></a><small><b>Aviso</b>: Esta aplicación tiene fines educativos y proporciona información básica sobre el reciclaje. En caso de duda, se recomienda consultar las fuentes mencionadas en los recursos adicionales para obtener información más detallada y precisa.</small>', unsafe_allow_html=True)
