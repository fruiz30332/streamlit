# Preparación del entorno
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
import streamlit as st
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

# funcion para realizar y almacenar el entrenamiento del modelo

def entrenamiento():
    # Carga del conjunto de datos MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizar los datos de entrada
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Convertir las etiquetas a formato categórico
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    # Definición del modelo de red neuronal
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Entrenamiento del modelo
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    # Evaluación del modelo
    score = model.evaluate(X_test, y_test, verbose=0)
    #print("Precisión del modelo:", score[1])
    st.write("Precisión del modelo:", score[1])

    # Guardar el modelo en formato HDF5
    model.save('modelo_entrenado.h5')
 


# Diseño de la interfaz de usuario con Streamlit


# Ruta del archivo del modelo
ruta_archivo = 'modelo_entrenado.h5'
#ruta_archivo = 'modelo_entrenado.keras'

# Verificar si el archivo del modelo existe 
if os.path.exists(ruta_archivo):
    # Cargar el modelo si existe
    modelo = tf.keras.models.load_model(ruta_archivo)
    print("Modelo cargado correctamente")
else:
    # crear el modelo
    entrenamiento()
    # Cargar el modelo 
    modelo = tf.keras.models.load_model(ruta_archivo)



# Diseño de la interfaz de usuario
st.title("Aplicación de reconocimiento de dígitos manuscritos")

# Área para cargar la imagen del usuario
imagen_cargada = st.file_uploader("Cargar imagen")

# Función para mostrar la imagen cargada
def mostrar_imagen(imagen):
    if imagen is not None:
        st.image(imagen, caption="Imagen cargada")

# Mostrar la imagen cargada
mostrar_imagen(imagen_cargada)

# Área de dibujo para el usuario (opcional)
dibujo = st.empty()

# Función para convertir el dibujo del usuario en un array de entrada para el modelo
def preprocesar_dibujo(dibujo):
    imagen = np.array(dibujo.to_image()).astype('float32')
    imagen = imagen / 255.0
    imagen = imagen.reshape(1, 28, 28, 1)
    return imagen

# Función para cargar una imagen
def cargar_imagen(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    imagen = np.array(imagen) / 255.0
    imagen = imagen.reshape(1, 28, 28, 1)
    return imagen

# Botón para realizar la predicción
boton_predecir = st.button("Predecir")

if boton_predecir:
    # Si se ha cargado una imagen, usarla como entrada para la predicción
    if imagen_cargada is not None:
        imagen_usuario = cargar_imagen(imagen_cargada.name)

    # Si no se ha cargado una imagen, usar el dibujo del usuario (opcional)
    else:
        imagen_usuario = preprocesar_dibujo(dibujo)

    # Realizar la predicción
    prediccion = modelo.predict(imagen_usuario)
    clase_predicha = np.argmax(prediccion)

    # Mostrar la predicción
    st.write("Dígito predicho:", clase_predicha)


