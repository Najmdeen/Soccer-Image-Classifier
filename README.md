# Soccer Player Image Classifier

## Introduction
This project is a sophisticated image classifier that uses both Support Vector Machine (SVM) and Convolutional Neural Network (CNN) models to identify images of famous soccer players. The application, built with Streamlit, allows users to upload an image of a player and choose between the SVM and CNN classifiers for prediction. The eligible players for identification in this project are Lionel Messi, Cristiano Ronaldo, Kylian Mbappé, Asisat Oshoala, and Alex Morgan.

## Features
- **Dual Classification Models:** Utilizes both SVM and CNN for image classification.
- **Streamlit Web App:** An interactive user interface for uploading images and displaying predictions.
- **Player Identification:** Capable of identifying five famous soccer players from uploaded images.

## Installation
To run this project, you need to install the required libraries. Use the following command:

```shell
pip install -r requirements.txt
```

## Usage
To start the Streamlit application, navigate to the project directory and run:
```shell
streamlit run main.py
```
## Project Structure
- **'cnn_classifier.ipynb:'** Jupyter notebook for building and training the CNN model.
- **'image_classifier.ipynb:'** Jupyter notebook for building and training the SVM model.
- **'main.py:'** The main Streamlit application for the interactive web interface.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.
