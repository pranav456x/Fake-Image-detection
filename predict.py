import numpy as np
import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import glob

model = load_model('/Users/pranav/Desktop/deepfake_detection.h5')


def preprocess_image(image_path):
    
    image = cv2.imread(image_path)
    
    
    if image is None:
        raise ValueError(f"Error: Could not load image at path '{image_path}'. Please check the file path or the image file.")
    
    
    image = cv2.resize(image, (96, 96))
    
    
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  
    
    return image


def predict_image(image_path):
    try:
        image = preprocess_image(image_path)
        prediction = model.predict(image)[0][0]  
        return "Fake" if prediction < 0.5 else "Real"  
    except Exception as e:
        return f"Prediction error: {str(e)}"

#testing folder given below
image_directory = "/Users/pranav/Desktop/j"

# Load image paths using glob
image_paths = glob.glob(os.path.join(image_directory, "*.jpg"))

if image_paths:
    result = predict_image(image_paths[0])  
    print(f"The image is {result}")
else:
    print("No images found.")
