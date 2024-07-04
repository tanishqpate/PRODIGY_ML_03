import os
import numpy as np
import cv2
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input

cat_folder = 'cats'
dog_folder = 'dogs'

IMG_SIZE = (128, 128)

def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(label)
    return images, labels

cat_images, cat_labels = load_images(cat_folder, 0)
dog_images, dog_labels = load_images(dog_folder, 1)

X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def extract_features_vgg16(imgs):
    model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    features = []
    for img in imgs:
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = model.predict(img)
        features.append(feature.flatten())
    return np.array(features)

X_train_features = extract_features_vgg16(X_train)
X_test_features = extract_features_vgg16(X_test)

svm = SVC(kernel='linear')
svm.fit(X_train_features, y_train)

with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

model = VGG16(weights='imagenet', include_top=False)
model.save('vgg16_model.h5')

y_pred = svm.predict(X_test_features)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
