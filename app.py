import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained model
model = load_model('model_Covid_19.h5')

# Define a function to detect Covid-19 from a chest X-ray image
def detect_covid(image):
    # Resize the image to 224x224 pixels
    resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    # Normalize the image pixel values to be between 0 and 1
    normalized = resized / 255.0
    # Add an extra dimension to the image array to match the model input shape
    expanded = np.expand_dims(normalized, axis=0)
    # Make a prediction with the model
    prediction = model.predict(expanded)
    # Get the predicted class label
    if prediction[0] >= 0.5:
        label = 'Covid-19'
    else:
        label = 'Normal'
    # Return the predicted class label
    return label

# Define the Streamlit app
def app():
    st.title("Covid-19 Detection Using Chest X-Ray")
    st.write("Upload a chest X-ray image to detect Covid-19")
    # Allow the user to upload an image file
    image_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        # Read the image file as a numpy array
        image = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        # Decode the image to BGR format
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # Display the image
        st.image(image, channels='BGR', caption='Uploaded Image')
        # Detect the Covid-19 status of the image
        label = detect_covid(image)
        # Display the result
        st.write("Covid-19 Status:", label)

if __name__ == '__main__':
    app()
