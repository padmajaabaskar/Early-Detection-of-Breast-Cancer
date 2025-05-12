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

def classify_tumor_normal(image_path, model, brightness_threshold, intensity_threshold):
    brightness = compute_brightness(image_path)
    intensity = compute_intensity(image_path)

    if brightness < brightness_threshold and intensity < intensity_threshold:
        return 'MALIGNANT CELL'
    else:
        return 'BENIGN CELL'

def compute_brightness(image_path):
    img = Image.open(image_path)
    img_gray = img.convert('L')
    brightness = np.mean(img_gray)
    brightness /= 50
    return brightness

def compute_intensity(image_path):
    img = Image.open(image_path)
    img_gray = img.convert('L')
    intensity = np.mean(img_gray)
    intensity /= 40
    return intensity

image_path = '/content/drive/MyDrive/MLO_ROI_HII/high_intensity_region_15.jpg'

brightness_threshold = 20
intensity_threshold = 30

classification = classify_tumor_normal(image_path, model, brightness_threshold, intensity_threshold)
print("Classification:", classification)