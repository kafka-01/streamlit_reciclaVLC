import streamlit as st
import folium
from streamlit_folium import folium_static

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

# Desplegable de los 88 barrios de Valencia
# Lista de los 88 barrios de Valencia
barrios = ['Algirós', 'Benicalap', 'Benimaclet', 'Camins al Grau', 'Campanar', 'Ciutat Vella', 'El Pla del Real', 'Extramurs', 'Jesús', 'L\'Olivereta', 'La Saïdia', 'Patraix', 'Poblats Marítims', 'Quatre Carreres', 'Rascanya', 'Alboraia', 'Albuixech', 'Alfara del Patriarca', 'Almàssera', 'Bonrepòs i Mirambell', 'Burjassot', 'Emperador', 'Godella', 'La Pobla de Farnals', 'Mislata', 'Moncada', 'Nàquera', 'Rocafort', 'Tavernes Blanques', 'Vinalesa', 'Xirivella', 'Xàtiva', 'Albal', 'Alcàsser', 'Aldaia', 'Alfafar', 'Benetússer', 'Beniparrell', 'Catarroja', 'Massanassa', 'Picanya', 'Picassent', 'Sedaví', 'Silla', 'Torrent', 'El Puig de Santa Maria', 'Faura', 'Massamagrell', 'Meliana', 'Museros', 'Puzol', 'Rafelbunyol', 'Sagunt', 'Albalat dels Sorells', 'Alboraya', 'Albuixech', 'Alfara del Patriarca', 'Almàssera', 'Benifairó de les Valls', 'Bonrepòs i Mirambell', 'Burjassot', 'Foios', 'Godella', 'La Pobla de Farnals', 'Massalfassar', 'Massamagrell', 'Meliana', 'Moncada', 'Museros', 'Nàquera', 'Pobla de Vallbona (la)', 'Puçol', 'Rafelbunyol', 'Rocafort', 'Sagunt', 'Tavernes Blanques', 'Tavernes de la Valldigna', 'València', 'Vinalesa']
barrio_seleccionado = st.selectbox('Selecciona tu barrio', barrios)

# Mostrar mapa de Valencia

# Crear un mapa centrado en Valencia
valencia_map = folium.Map(location=[39.46975, -0.37739], zoom_start=12)

# Añadir un marcador en Valencia
folium.Marker(location=[39.46975, -0.37739], popup="Valencia").add_to(valencia_map)

# Mostrar el mapa en Streamlit
folium_static(valencia_map)


st.markdown('''
## Identificar los residuos

Aprende a reconocer y clasificar los distintos tipos de residuos comunes, como orgánicos, papel y cartón, vidrio, plástico, metal, etc. Esto te ayudará a saber en qué contenedor depositar cada tipo de residuo.

Si tienes alguna duda sobre cómo reciclar un objeto específico, no dudes en utilizar nuestra herramienta de Identificación de Residuos. Sube una foto con fondo blanco del objeto en cuestión, y te ayudaremos a determinar en qué contenedor debe ser depositado.<sup><a name="back" href="#disclaimer">[1]</a></sup>
''', unsafe_allow_html=True)

# Formulario para subir una imagen
st.subheader('Identificación de los Residuos')
imagen = st.file_uploader('Sube una imagen con fondo blanco del objeto a identificar', type=['jpg', 'jpeg', 'png'])
if imagen is not None:
    # Mostrar la imagen si es válida
    st.image(imagen)

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
