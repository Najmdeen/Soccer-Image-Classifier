import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import joblib  # For loading non-Keras models like SVM
import pywt


def crop_face_and_eyes(image):
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:  # Only save if two eyes are detected
            cropped_face = img[y : y + h, x : x + w]
            return cropped_face


def wavelet_transform_on_image(image):
    # Perform 2D Discrete Wavelet Transform
    coeffs2 = pywt.dwt2(image, "haar")
    LL, _ = coeffs2

    # Define the common size for all images
    common_size = (100, 100)

    # Resize, flatten, and convert X to float, then create a NumPy array
    X = np.array(cv2.resize(LL, common_size).flatten().astype(float))
    return X


@st.cache(allow_output_mutation=True)
def load_my_model(model_choice):
    if model_choice == "CNN":
        return load_model("best_model.h5")
    elif model_choice == "SVM":
        return joblib.load("svm_pipeline_model.joblib")


model_choice = st.sidebar.selectbox("Choose a model:", ("CNN", "SVM"))
model = load_my_model(model_choice)


st.title("Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# Display uploaded Image
st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

if uploaded_file is not None:
    img = crop_face_and_eyes(uploaded_file)

    if img is not None:
        if model_choice == "CNN":
            img = cv2.resize(img, (100, 100)) / 255.0
            img = img.reshape(1, 100, 100, 3)
        elif model_choice == "SVM":
            img = wavelet_transform_on_image(img)

        if st.sidebar.button("Predict"):
            predicted_label = None
            probabilities = None

            if model_choice == "CNN":
                #   # Predict probabilities
                prediction = model.predict(img)
                probabilities = prediction[0]
                # get the max value in the predictions probabilities
                predicted = np.argmax(prediction, axis=1)
                predicted_label = predicted[0]

            elif model_choice == "SVM":
                # Predict image label
                predicted_label = model.predict(img.reshape(1, -1))[0]
                # Predict probabilities
                probabilities = model.predict_proba(img.reshape(1, -1))[0]

            # print the predicted label
            players = [
                "Asisat Oshoala",
                "Kylian Mbappe",
                "Cristiano Ronaldo",
                "Lionel Messi",
                "Alex Morgan",
            ]
            probability_per_player = {
                player: prob for player, prob in zip(players, probabilities)
            }

            # Display the label with its corresponding probability and predicted label
            st.write(f"Player is: {players[predicted_label]}")
            st.header("Probabilities of predicted label")
            for player, prob in probability_per_player.items():
                st.write(f"{player}: {prob:.2f}")
    else:
        st.write("No face with two eyes detected.")
