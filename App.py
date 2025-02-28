import streamlit as st
import numpy as np
import cv2
import random
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('/Users/pranav/Desktop/deepfake_detection.h5')

# Preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (96, 96))  # Resize to the model's expected input size
    image = img_to_array(image)  # Convert image to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Predict if the image is fake or real
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    st.write("Prediction probabilities:", prediction)  # Debugging: Display raw prediction
    if prediction.shape[1] == 1:  # Binary classification with single output
        class_label = 1 if prediction[0][0] > 0.5 else 0
    else:  # Multiclass classification
        class_label = np.argmax(prediction, axis=1)[0]
    return "Fake" if class_label == 0 else "Real"

# Random messages for Fake and Real classifications
FAKE_MESSAGES = [
    "This image seems to be fake. AI detected subtle inconsistencies typically present in deepfakes.",
    "Our analysis suggests this is a fake image. Certain artifacts or irregularities have been identified.",
    "This appears to be a deepfake. Anomalies such as mismatched lighting or unnatural features were detected."
]






REAL_MESSAGES = [
    "This image is classified as real. No significant signs of manipulation were found.",
    "AI confirms this is a real image. It lacks the typical inconsistencies seen in deepfakes.",
    "This is identified as a real image. The analysis did not detect any features of forgery."
]

# Apply custom font for the title using Azonix
st.markdown(
    """
    <style>
        @font-face {
            font-family: 'Azonix';
            src: url('Azonix.ttf');  /* Replace with the path to your Azonix font */
        }
        .custom-title {
            font-family: 'Azonix', sans-serif;
            text-align: center;
            color: grey;
            font-size: 40px;
        }
    </style>
    """, unsafe_allow_html=True
)

# Title with the custom font
st.markdown("<h1 class='custom-title'>DEEP FAKE IMAGE DETECTION</h1>", unsafe_allow_html=True)

# Streamlit application
# Add a header image with reduced size and centered
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column ratios for centering
with col2:
    st.image("/Users/pranav/Desktop/IMG_1265.JPG", use_container_width=True)

# Detailed description about deepfake
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # To read file as bytes:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image")

    # Predict and display result
    result = predict_image(image)

    # Set the color and description based on the result
    if result == "Fake":
        color = "red"
        description = random.choice(FAKE_MESSAGES)  # Pick a random fake message
    else:
        color = "green"
        description = random.choice(REAL_MESSAGES)  # Pick a random real message

    # Display the title with the appropriate color
    st.markdown(f"<h1 style='color:{color};'>{result} Image</h1>", unsafe_allow_html=True)

    # Display the description
    st.write(description)

# Understanding deepfakes section
st.header("Understanding Deepfakes")
st.write("""
Deepfakes are synthetic media where a person in an existing image or video is replaced with someone else's likeness. Leveraging sophisticated AI algorithms, primarily deep learning techniques, deepfakes can create incredibly realistic and convincing fake videos and images. This technology, while having legitimate uses in entertainment and education, poses significant ethical and security challenges. Deepfakes can be used to spread misinformation, create malicious content, and impersonate individuals without consent, raising serious concerns about privacy and trust in digital media. Detection of deepfakes is crucial to mitigate these risks, and AI plays a vital role in identifying such manipulations. By analyzing subtle artifacts and inconsistencies that are often imperceptible to the human eye, AI models can effectively distinguish between real and fake media, ensuring the integrity of visual content.
""")


