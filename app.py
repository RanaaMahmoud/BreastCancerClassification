import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from datetime import datetime

# Load the model once
@st.cache_resource
def load_model():
    model_path = "model.h5"
    return tf.keras.models.load_model(model_path)

model = load_model()

# Optimized inference
@tf.function(input_signature=[tf.TensorSpec(shape=[1, 150, 150, 3], dtype=tf.float32)])
def predict_inference(img_array):
    return model(img_array)

# Image preprocessing
def preprocess_image(uploaded_img, target_size=(150, 150)):
    img = uploaded_img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Streamlit UI
st.set_page_config(page_title="🩺 Breast Cancer Detection", layout="centered")
st.title("🧬 Breast Cancer Classifier")
st.markdown("Upload a mammogram image to check if it is **Benign** or **Malignant**.")

uploaded_file = st.file_uploader("📤 Upload a mammogram image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Classifying..."):
        img_array = preprocess_image(pil_image)
        prediction = predict_inference(img_array)
        prob = prediction[0][0].numpy()
        label = "Malignant" if prob > 0.5 else "Benign"
        confidence = prob if prob > 0.5 else 1 - prob

    st.success(f"🩻 **Prediction:** {label}")
    st.info(f"📊 **Confidence:** {confidence*100:.2f}%")

    # Save result if needed
    if st.checkbox("💾 Save this result"):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_filename = f"result_{timestamp}.txt"
        with open(result_filename, 'w') as f:
            f.write(f"Prediction: {label}, Confidence: {confidence:.2f}")
        st.success("✅ Saved locally")

else:
    st.info("👆 Please upload a mammogram image to begin.")
