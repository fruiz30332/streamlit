import streamlit as st
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_cat
import cv2

# Carga del conjunto de datos MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar los datos de entrada
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convertir las etiquetas a formato categórico
y_train = to_cat(y_train, num_classes=10)
y_test = to_cat(y_test, num_classes=10)

# Definición del modelo de red neuronal
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Evaluación del modelo
score = model.evaluate(X_test, y_test, verbose=0)
print("Precisión del modelo:", score[1])

# Diseño de la interfaz de usuario con Streamlit
st.title("Aplicación de reconocimiento de dígitos manuscritos")

# Área para cargar la imagen del usuario
imagen_cargada = st.file_uploader("Cargar imagen")

# Función para mostrar la imagen cargada
def mostrar_imagen(imagen):
    if imagen is not None:
        st.image(imagen, caption="Imagen cargada")

# Mostrar la imagen cargada
mostrar_imagen(imagen_cargada)

# Área de dibujo para el usuario
dibujo = st.empty()

# Función para convertir el dibujo del usuario en un array de entrada para el modelo
def preprocesar_dibujo(dibujo):
    imagen = np.array(dibujo.to_image()).astype('float32')
    imagen = imagen / 255.0
    imagen = imagen.reshape(1, 28, 28, 1)
    return imagen

# Botón para realizar la predicción
boton_predecir = st.button("Predecir")

if boton_predecir:
    # Si se ha cargado una imagen, usarla como entrada para la predicción
    if imagen_cargada is not None:
        imagen_usuario = cargar_imagen(imagen_cargada.name)

    # Si no se ha cargado una imagen, usar el dibujo del usuario
    else:
        imagen_usuario = preprocesar_dibujo(dibujo)

    # Realizar la predicción
    prediccion = model.predict(imagen_usuario)
    clase_predicha = np.argmax(prediccion)

    # Mostrar la predicción
    st.write("Dígito predicho:", clase_predicha)

# Función para cargar una imagen
def cargar_imagen(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    imagen = np.array(imagen) / 255.0
    imagen = imagen.reshape(1, 28, 28, 1)
    return imagen

if __name__ == "__main__":
    st.run()
