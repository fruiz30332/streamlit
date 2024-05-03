import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST model (replace with your actual path)
model = tf.keras.models.load_model("path/to/model.h5")

def predict_digit(image_file):
  # Read the uploaded image
  img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
  
  # Preprocess the image
  img = cv2.resize(img, (28, 28))
  img = img.astype('float32') / 255.0
  img_data = np.expand_dims(img, axis=0)
  
  # Make predictions
  predictions = model.predict(img_data)
  predicted_digit = np.argmax(predictions[0])
  return predicted_digit


# Title and description for the app
st.title("MNIST Digit Prediction")
st.write("Upload an image of a handwritten digit (0-9) for prediction.")
  
# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
  
if uploaded_file is not None:
    # Display uploaded image (optional)
    st.image(cv2.cvtColor(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    
    # Make prediction and display result
    predicted_digit = predict_digit(uploaded_file)
    st.write(f"Predicted digit: {predicted_digit}")


