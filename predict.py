import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Image dimensions
IMG_SIZE = (128, 128)

# Load the trained SVM model
with open('svm_model.pkl', 'rb') as f:
    svm = pickle.load(f)

# Load the feature extraction model
vgg16_model = load_model('vgg16_model.h5')

# Function to predict a single image
def predict_image(img_path):
    # Load the image
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, IMG_SIZE)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Extract features using VGG16
        features = vgg16_model.predict(img)
        features = features.flatten().reshape(1, -1)

        # Predict using the trained SVM
        prediction = svm.predict(features)
        if prediction[0] == 0:
            print("There is cat in image.")
        else:
            print("There is dog in image.")
    else:
        print("Image not loaded correctly.")

# Example usage
img_path = 'testimage1.jpeg'
predict_image(img_path)
