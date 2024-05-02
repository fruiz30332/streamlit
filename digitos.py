import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Cargar el modelo preentrenado de Keras (MNIST)
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)

# Función para preprocesar la imagen
def preprocess_image(image):
    # Convertir a escala de grises
    image = image.convert('L')
    # Redimensionar a 28x28 píxeles
    image = image.resize((28, 28))
    # Normalizar los valores de píxeles
    image_array = np.array(image) / 255.0
    # Agregar un tercer canal para que coincida con las expectativas del modelo
    image_array = np.expand_dims(image_array, axis=-1)
    return image_array

# Widget para cargar una imagen
uploaded_image = st.file_uploader("Cargar una imagen de un dígito manuscrito", type=["jpg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesar la imagen
    preprocessed_image = preprocess_image(image)
    st.image(preprocessed_image, caption="Imagen procesadada", use_column_width=True)
    # Hacer predicciones con el modelo
    predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
    predicted_class = np.argmax(predictions)

    st.write(f"Clase predicha: {predicted_class}")
