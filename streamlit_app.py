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
labels = ['Papel / Carton', 'Vidrio', 'Ecoparque móvil', 'Papel / Carton', 'Envases Ligeros', 'Residuos Urbanos']
    
# Secret
my_email = st.secrets['email']
model_weight_file = st.secrets['model_url']    
    
# Load the model into cache at the beginning of execution
@st.cache_resource(show_spinner = False)
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

@st.cache_data(ttl = "1h", show_spinner = False)
def get_neighborhoods():
    url = "https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=barris-barrios&q=&rows=-1"
    response = requests.get(url)
    data = response.json()

    neighborhoods_json = data["records"]

    # Sort the records by the 'nombre' field
    neighborhoods_json = sorted(neighborhoods_json, key=lambda x: x['fields']['nombre'])

    neighborhoods = []

    for neighborhood in neighborhoods_json:
        neighborhoods.append({
            "name": ' '.join(word.capitalize() for word in neighborhood["fields"]["nombre"].split()),
            "geo_shape": neighborhood["fields"]["geo_shape"]
        })

    
    neighborhoods_loaded = True
    return neighborhoods

@st.cache_data(ttl = "1h", show_spinner = False)
def get_containers(neighborhood_shape):
    coordinates = neighborhood_shape['coordinates'][0]  # Get the list of coordinates of the polygon

    # Format the coordinates of the polygon
    coordinates_str = '%2C+'.join([f'({coord[1]}%2C+{coord[0]})' for coord in coordinates])

    url_solid_waste = f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=contenidors-residus-solids-contenidores-residuos-solidos&q=&rows=-1&geofilter.polygon=" + coordinates_str
    url_glass = f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=contenidors-vidre-contenedores-vidrio&q=&rows=-1&geofilter.polygon=" + coordinates_str

    response = requests.get(url_solid_waste)
    data = response.json()
    combined_results = data['records']

    response = requests.get(url_glass)
    data = response.json()
    glass_records = data['records']
    for record in glass_records:
        record['fields']['tipo_resid'] = 'Vidrio'
    combined_results += glass_records
    
    return combined_results

@st.cache_data(ttl = "1h", show_spinner = False)
def get_containers2(neighborhood_shape):
    coordinates = neighborhood_shape['coordinates'][0]  # Get the list of coordinates of the polygon

    # Format the coordinates of the polygon
    coordinates_str = '%2C+'.join([f'({coord[1]}%2C+{coord[0]})' for coord in coordinates])

    url_batteries = f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=localitzacio-contenidors-piles-localizacion-contenedores-pilas&q=&rows=-1&geofilter.polygon=" + coordinates_str
    url_used_oil = f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=contenidors-oli-usat-contenedores-aceite-usado&q=&rows=-1&geofilter.polygon=" + coordinates_str
    url_mobile_ecoparks = f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=ecoparcs-mobils-ecoparques-moviles&q=&rows=-1&geofilter.polygon=" + coordinates_str
    url_clothes = f"https://valencia.opendatasoft.com/api/records/1.0/search/?dataset=contenidors-de-roba-contenedores-de-ropa&q=&rows=-1&geofilter.polygon=" + coordinates_str

    response = requests.get(url_batteries)
    data = response.json()
    combined_results = data['records']
    for record in combined_results:
        record['fields']['tipo_resid'] = 'Pilas'

    response = requests.get(url_used_oil)
    data = response.json()
    used_oil_records = data['records']
    for record in used_oil_records:
        record['fields']['tipo_resid'] = 'Aceite usado'
    combined_results += used_oil_records

    response = requests.get(url_mobile_ecoparks)
    data = response.json()
    mobile_ecoparks_records = data['records']
    for record in mobile_ecoparks_records:
        record['fields']['tipo_resid'] = 'Ecoparque móvil'
    combined_results += mobile_ecoparks_records

    response = requests.get(url_clothes)
    data = response.json()
    clothes_records = data['records']
    for record in clothes_records:
        record['fields']['tipo_resid'] = 'Ropa'
    combined_results += clothes_records

    return combined_results

@st.cache_resource(show_spinner = False)
def generar_mapa(center, zoom, containers):
    
    feature_group = folium.FeatureGroup(name="Markers")  # Create a FeatureGroup for markers        
        
    # Iterate over each key and add markers
    for container_type, coordinates in containers.items():
        for coords in coordinates:
            # Get the coordinates of the point
            lat = coords["coordinates"][1]
            lon = coords["coordinates"][0]

            # Create a marker with a Folium compatible icon (e.g., "leaf") and the corresponding color
            icon = folium.CustomIcon(icon_image=get_icon(container_type), icon_size=(42, 36))
            marker = folium.Marker(location=[lat, lon], popup=container_type.replace(" / ", "/"), icon=icon)

            # Add the marker to the map
            feature_group.add_child(marker)  # Add the marker to the feature group
            
    valencia_map = folium.Map(location=center, zoom_start=zoom, width=500, height=700)
    valencia_map.add_child(feature_group)
    return valencia_map

def calculate_center_zoom(neighborhood_shape):
    coords_neighborhood = neighborhood_shape["coordinates"][0]

    latitudes = [coord[1] for coord in coords_neighborhood]
    longitudes = [coord[0] for coord in coords_neighborhood]

    min_lat = min(latitudes)
    max_lat = max(latitudes)
    min_lon = min(longitudes)
    max_lon = max(longitudes)

    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    zoom = 15  # Initial zoom level

    lat_extent = max_lat - min_lat
    lon_extent = max_lon - min_lon

    if lat_extent != 0 and lon_extent != 0:
        max_zoom = 18  # Maximum allowed zoom (adjust according to your needs)
        width_pixels = 500  # Map width in pixels (adjust according to your needs)
        deg_per_pixel = (lon_extent / width_pixels)  # Degrees of longitude per pixel
        zoom = math.floor(math.log((360 * math.cos(math.radians(center_lat))) / deg_per_pixel, 2))
        zoom = min(max_zoom, zoom)

    return (center_lat, center_lon), zoom - 2
        
def locate_containers_app():
    
    # Form

    neighborhoods = get_neighborhoods()
    
    neighborhoods_list = [nh["name"] for nh in neighborhoods]

    selected_neighborhood = st.selectbox('Selecciona tu barrio', neighborhoods_list)
    
    # Get the shape of the selected neighborhood
    nh_index = neighborhoods_list.index(selected_neighborhood)
    nh_shape = neighborhoods[nh_index]['geo_shape']

    # List of container types
    container_types = ['Residuos sólidos', 'Aceite usado', 'Pilas', 'Ecoparque móvil', 'Ropa']

    # Widget selectbox to select the container type
    selected_container = st.selectbox("Selecciona el tipo de contenedor", container_types)

    if (selected_container == 'Residuos sólidos'):

        # List of waste types
        waste_types = ['Residuos Urbanos', 'Envases Ligeros', 'Organico', 'Papel / Carton', 'Vidrio', 'Todos']

        # Widget selectbox to select the waste type
        selected_waste = st.selectbox("Selecciona el tipo de residuo", waste_types)

    else:

        selected_waste = ''

    with st.spinner("Refreshing Map..."):
        
        # Proccess
    
        selected_neighborhood = neighborhoods[nh_index]['name']
        nh_shape = neighborhoods[nh_index]['geo_shape']
        
        # Show map of Valencia
    
        # Calculate the optimal central location and zoom level
        center, zoom = calculate_center_zoom(nh_shape)
    
        containers = {}        
    
        if (selected_container == 'Residuos sólidos'):            
            containers1_json = get_containers(nh_shape)
            containers_json = containers1_json
            selection = selected_waste
        else:
            containers2_json = get_containers2(nh_shape)
            containers_json = containers2_json
            selection = selected_container
            
        counter = 0
    
        for c in containers_json:
            container_type = c["fields"]["tipo_resid"]
            if (container_type == selection or selection == 'Todos'):
                coords = c["fields"]["geo_shape"]
                # Check if the key exists in the dictionary
                if container_type not in containers:
                    containers[container_type] = []
    
                # Add the coordinates to the corresponding list
                containers[container_type].append(coords)
                
                counter += 1
    
        valencia_map = generar_mapa(center, zoom, containers)
        folium_static(valencia_map)
    st.write(f"Contenedores encontrados: {counter}")

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

## Localización de los contenedores cercanos
Utiliza nuestra herramienta interactiva para encontrar los contenedores más cercanos a tu barrio. Selecciona tu barrio en el menú desplegable y podrás visualizar en el mapa los contenedores específicos. ¡Recuerda seguir las indicaciones y depositar los residuos en el contenedor correcto!
"""
st.markdown(intro_text)

# locate containers
with st.container():    
        
    st.image(img_reciclaVLC_app1, width=64)     
    locate_containers_app()
    
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
