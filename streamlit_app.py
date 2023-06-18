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

# Get the absolute path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))

# Open the image files
img_reciclaVLC = Image.open(os.path.join(current_path, 'images', 'ReciclaVLC.png'))
img_reciclaVLC_app1 = Image.open(os.path.join(current_path, 'images', 'ReciclaVLC-Localiza.png'))
img_reciclaVLC_app2 = Image.open(os.path.join(current_path, 'images', 'ReciclaVLC-Identifica.png'))

def obtener_icono(label):
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

# IDENTIFICACIÓN DE RESIDUOS ============================================================================================

# Variable de bandera para verificar si el modelo ha sido cargado
modelo_cargado = False

# Secreto
my_email = st.secrets['email']
model_weight_file = st.secrets['model_url']
    
#Descargar el modelo inicial

model_path = os.path.join(current_path, 'models', 'model.h5')

if not os.path.exists('./models/model.h5'):
    u = urlopen(model_weight_file)
    data = u.read()
    u.close()
    with open(model_path, 'wb') as f:
        f.write(data)

# Cargar el modelo en caché al principio de la ejecución
@st.cache_resource 
def cargar_modelo():
    model = load_model(model_path)
    return model

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

# LOCALIZACION DE CONTENEDORES =================================================================================

# Variable de bandera para verificar si la lista de barrios ha sido cargada
barrios_cargado = False

# Variable de bandera para verificar si los contenedores de residuos solidos + vidrio del barrio ya han sido cargados
contenedores1_cargado = None

# Variable de bandera para verificar si los contenedores pilas, aceite y ecoparques moviles del barrio ya han sido cargados
contenedores2_cargado = None

@st.cache_data
def get_barrios():
    url = "https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=barris-barrios&q=&rows=-1"
    response = requests.get(url)
    data = response.json()
    return data["records"]

@st.cache_data
def get_contenedores(filtro):
    coordinates = filtro['coordinates'][0]  # Obtener la lista de coordenadas del polígono

    # Formatear las coordenadas del polígono
    coordinates_str = '%2C+'.join([f'({coord[1]}%2C+{coord[0]})' for coord in coordinates])

    url_residuos_solidos = f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=contenidors-residus-solids-contenidores-residuos-solidos&q=&rows=-1&geofilter.polygon=" + coordinates_str
    url_vidrio = f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=contenidors-vidre-contenedores-vidrio&q=&rows=-1&geofilter.polygon=" + coordinates_str

    response = requests.get(url_residuos_solidos)
    data = response.json()
    combined_results = data['records']

    response = requests.get(url_vidrio)
    data = response.json()
    vidrio_records = data['records']
    for record in vidrio_records:
        record['fields']['tipo_resid'] = 'Vidrio'
    combined_results += vidrio_records

    return combined_results

@st.cache_data
def get_contenedores2(filtro):
    coordinates = filtro['coordinates'][0]  # Obtener la lista de coordenadas del polígono

    # Formatear las coordenadas del polígono
    coordinates_str = '%2C+'.join([f'({coord[1]}%2C+{coord[0]})' for coord in coordinates])

    url_pilas = f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=localitzacio-contenidors-piles-localizacion-contenedores-pilas&q=&rows=-1&geofilter.polygon=" + coordinates_str
    url_aceite = f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=contenidors-oli-usat-contenedores-aceite-usado&q=&rows=-1&geofilter.polygon=" + coordinates_str
    url_ecoparques_moviles = f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=ecoparcs-mobils-ecoparques-moviles&q=&rows=-1&geofilter.polygon=" + coordinates_str

    response = requests.get(url_pilas)
    data = response.json()
    combined_results = data['records']
    for record in combined_results:
        record['fields']['tipo_resid'] = 'Pilas'

    response = requests.get(url_aceite)
    data = response.json()
    aceite_records = data['records']
    for record in aceite_records:
        record['fields']['tipo_resid'] = 'Aceite usado'
    combined_results += aceite_records

    response = requests.get(url_ecoparques_moviles)
    data = response.json()
    ecoparques_moviles_records = data['records']
    for record in ecoparques_moviles_records:
        record['fields']['tipo_resid'] = 'Ecoparque móvil'
    combined_results += ecoparques_moviles_records

    return combined_results

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

def obtener_texto_seguridad(prob):
    if prob < 0.3:
        return "Hmm... parece que la clasificación no está del todo clara. Puede que necesitemos más datos o un ajuste en el modelo. Sigamos trabajando para mejorar."
    elif prob < 0.6:
        return "La clasificación es relativamente segura, aunque aún existe un margen de error. Sigamos refinando nuestros conocimientos en el reciclaje para ofrecer resultados más precisos."
    elif prob < 0.9:
        return "¡Excelente! La clasificación es bastante segura, lo cual es alentador. Nuestra dedicación al reciclaje y la conciencia ecológica están dando frutos."
    else:
        return "¡Increíble! La clasificación es muy segura. Nuestro compromiso con el reciclaje y la protección del medio ambiente está dando resultados notables. Sigamos cuidando de nuestro planeta."


# PÁGINA ======================================================================================================


# Logo
st.image(img_reciclaVLC)

# Título
st.title("Recicla Valencia: Cuida tu ciudad, separa tu basura")

# Introducción
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

### Localización de los contenedores cercanos
Utiliza nuestra herramienta interactiva para encontrar los contenedores más cercanos a tu barrio. Selecciona tu barrio en el menú desplegable y podrás visualizar en el mapa los contenedores específicos. ¡Recuerda seguir las indicaciones y depositar los residuos en el contenedor correcto!

"""
# Localización de residuos
st.image(img_reciclaVLC_app1, width=64)

with st.container():
    if not barrios_cargado:
        #Cargar los barrios y sus shapes
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

        barrios_cargado = True

    barrio_seleccionado = st.selectbox('Selecciona tu barrio', barrios)
    
    # Obtener la forma del barrio seleccionado
    indice_barrio = barrios.index(barrio_seleccionado)
    shape_barrio = shapes[indice_barrio]

    # Lista de tipos de residuos
    tipos_contenedor = ['Residuos sólidos', 'Aceite usado', 'Pilas', 'Ecoparque móvil']

    # Widget selectbox para seleccionar el tipo de residuo
    contenedor_seleccionado = st.selectbox("Selecciona el tipo de contenedor", tipos_contenedor)

    if (contenedor_seleccionado == 'Residuos sólidos'):
        # Lista de tipos de residuos
        tipos_residuos = ['Todos', 'Envases Ligeros', 'Organico', 'Papel / Carton', 'Residuos Urbanos', 'Vidrio']

        # Widget selectbox para seleccionar el tipo de residuo
        residuo_seleccionado = st.selectbox("Selecciona el tipo de residuo", tipos_residuos)
    
    with st.spinner("Loading..."):

        contenedores = {}

        if (contenedor_seleccionado == 'Residuos sólidos'):
            if contenedores1_cargado != barrio_seleccionado:
                contenedores1_json = get_contenedores(shape_barrio)
                contenedor1_cargado = barrio_seleccionado
            contenedores_json = contenedores1_json
            seleccion = residuo_seleccionado
        else:
            if contenedores2_cargado != barrio_seleccionado:
                contenedores2_json = get_contenedores2(shape_barrio)
                contenedor2_cargado = barrio_seleccionado
            contenedores_json = contenedores2_json
            seleccion = contenedor_seleccionado

        for c in contenedores_json:
            tipo = c["fields"]["tipo_resid"]
            coords = c["fields"]["geo_shape"]

            # Verificar si la clave existe en el diccionario
            if tipo not in contenedores:
                contenedores[tipo] = []

            # Agregar las coordenadas a la lista correspondiente
            contenedores[tipo].append(coords)
        
        # Mostrar mapa de Valencia

        # Calcular la ubicación central y el nivel de zoom óptimos
        center, zoom = calculate_center_zoom(shape_barrio)

        # Crear el mapa centrado en el barrio con el nivel de zoom óptimo
        valencia_map = folium.Map(location=center, zoom_start=zoom)

        # Recorrer cada clave y agregar marcadores
        for tipo, coordenadas in contenedores.items():
            if (tipo == seleccion or seleccion == 'Todos'):                
                for coords in coordenadas:
                    # Obtener las coordenadas del punto
                    lat = coords["coordinates"][1]
                    lon = coords["coordinates"][0]

                    # Crear un marcador con un icono compatible con Folium (por ejemplo, "leaf") y el color correspondiente
                    icon = folium.CustomIcon(icon_image=obtener_icono(tipo), icon_size=(42, 36))
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

st.subheader('Identificación de los residuos')

# Identificación de residuos
st.image(img_reciclaVLC_app2, width=64)

with st.container():    
    uploaded_file = st.file_uploader('Sube una imagen con fondo blanco del objeto a identificar', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        with st.spinner("Loading..."):
            image = Image.open(uploaded_file)
            if not modelo_cargado:
                modelo = cargar_modelo()
                modelo_cargado = True
            clase_predicha, prob = prediccion(image, modelo)
            
            # Obtener el icono y el texto de seguridad
            icono = obtener_icono(clase_predicha)
            texto_seguridad = obtener_texto_seguridad(prob)

            # Mostrar la imagen y la clasificación
            col1, col2 = st.columns([2, 4])  # Dividir el espacio en dos columnas
            with col1:
                #st.image(image, width=250, caption=f'Clase predicha: {clase_predicha}, Probabilidad: {prob:.2f}')
                st.image(image)

            # Mostrar el icono y el texto de seguridad
            with col2:
                st.image(Image.open(icono), width=50)  # Mostrar el icono como imagen
                st.markdown(f"**{clase_predicha}**\n{texto_seguridad}")  # Mostrar el texto de seguridad con Markdown

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
