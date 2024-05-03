import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Cargar el modelo preentrenado de Keras (MNIST)
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)


# Función para preprocesar la imagen
def preprocess_image2(image):
    # Convertir a escala de grises
    image = image.convert('L')
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # Redimensionar a 28x28 píxeles
    img = cv2.resize(img, (28, 28))
    # Normalizar los valores de píxeles
    img = img.astype('float32') / 255.0
    # Agregar un tercer canal para que coincida con las expectativas del modelo
    img_data = np.expand_dims(img, axis=0)
    return img_data

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
    #preprocessed_image = preprocess_image(image)
    preprocessed_image = preprocess_image2(image)
    st.image(preprocessed_image, caption="Imagen procesadada", use_column_width=True)
    
    
    # Hacer predicciones con el modelo
    predictions = model.predict(img_data)
    predicted_digit = np.argmax(predictions[0])
    st.write(f"Predicted digit: {predicted_digit}")

    #predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
    #predicted_class = np.argmax(predictions)

    #st.write(f"Clase predicha: {predicted_class}")
