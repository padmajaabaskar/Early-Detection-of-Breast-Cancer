import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = ResNet50(weights='imagenet', include_top=False)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    features = model.predict(preprocessed_image)
    return features

def extract_brightness(image_path):
    img = Image.open(image_path)
    img_gray = img.convert('L')
    brightness = np.mean(img_gray)
    return brightness

def extract_intensity(image_path):
    img = Image.open(image_path)
    img_gray = img.convert('L')
    intensity = np.mean(img_gray)
    return intensity

def extract_features_labels(directory, brightness_threshold, intensity_threshold):
    features = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            resnet_features = extract_features(image_path, model)
            brightness = extract_brightness(image_path)
            intensity = extract_intensity(image_path)
            if brightness < brightness_threshold and intensity < intensity_threshold:
                labels.append(1)
            else:
                labels.append(0)
            features.append(resnet_features)
    return np.array(features), np.array(labels)

dataset_dir = "/content/drive/MyDrive/MLO_ROI_HII"

brightness_threshold = 2
intensity_threshold = 70

features, labels = extract_features_labels(dataset_dir, brightness_threshold, intensity_threshold)

features = features.reshape(features.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)