import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from PIL import Image

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False)

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract features using the ResNet50 model
def extract_features(image_path, model):
    preprocessed_image = preprocess_image(image_path)
    features = model.predict(preprocessed_image)
    return features

# Function to classify tumor or normal based on brightness and intensity thresholds
def classify_tumor_normal(image_path, model, brightness_threshold, intensity_threshold):
    brightness = compute_brightness(image_path)
    intensity = compute_intensity(image_path)

    if brightness < brightness_threshold and intensity < intensity_threshold:
        return 'tumor'
    else:
        return 'normal'

# Function to compute brightness of the image
def compute_brightness(image_path):
    img = Image.open(image_path)
    img_gray = img.convert('L')
    brightness = np.mean(img_gray)
    brightness /= 20.0  # You can adjust this scaling factor as needed
    return brightness

# Function to compute intensity of the image
def compute_intensity(image_path):
    img = Image.open(image_path)
    img_gray = img.convert('L')
    intensity = np.mean(img_gray)
    intensity /= 24.0  # You can adjust this scaling factor as needed
    return intensity

# Path to the image
image_path = '/content/drive/MyDrive/MLO_ROI_HII/high_intensity_region_30.jpg'

# Define your custom brightness and intensity thresholds
brightness_threshold = 20
intensity_threshold = 10

# Classify the image
classification = classify_tumor_normal(image_path, model, brightness_threshold, intensity_threshold)
print("Classification:", classification)