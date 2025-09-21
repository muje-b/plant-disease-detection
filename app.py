import streamlit as st
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from PIL import Image
import ollama
import os
import gdown
import requests
import hashlib
import h5py
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="Plant Disease Detection & AI Assistant",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------
# Model Configuration
# -------------------------------
MODEL_PATH = "plant_disease_vgg16_optimized.h5"
FILE_ID = "1IYZz7ibvzsngy0mFbsUFFFLxjbT7_0Bi"

# -------------------------------
# Helper Functions
# -------------------------------
def is_valid_h5_file(file_path):
    """Check if a file is a valid HDF5 file"""
    try:
        with h5py.File(file_path, 'r') as f:
            return True
    except:
        return False

# -------------------------------
# Download VGG16 model from Google Drive with verification
# -------------------------------
@st.cache_resource
def download_and_verify_model():
    """Download the model with verification"""
    download_needed = True
    model_status = "Not downloaded"
    
    # Check if file exists and is valid
    if os.path.exists(MODEL_PATH):
        if is_valid_h5_file(MODEL_PATH):
            st.success("Model already exists and is valid!")
            download_needed = False
            model_status = "Valid"
        else:
            st.warning("Model file exists but is corrupted. Redownloading...")
            os.remove(MODEL_PATH)
            model_status = "Corrupted"
    
    # Download if needed
    if download_needed:
        st.info("Downloading VGG16 model from Google Drive...")
        MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
        
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            
            # Verify the download
            if is_valid_h5_file(MODEL_PATH):
                st.success("Model downloaded and verified successfully!")
                model_status = "Downloaded and valid"
            else:
                st.error("Downloaded file is not a valid HDF5 file.")
                model_status = "Invalid file"
                return False
                
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
            model_status = f"Download error: {str(e)}"
            return False
    
    return True

# -------------------------------
# Load Plant Disease Classification Model (VGG16)
# -------------------------------
@st.cache_resource
def load_vgg_model():
    """Load the VGG16 model with error handling"""
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found.")
        return None
    
    try:
        if not is_valid_h5_file(MODEL_PATH):
            st.error("Model file is corrupted. Please try downloading again.")
            return None
            
        st.info("Loading VGG16 model...")
        
        # Try to load with standard approach first
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH)
            st.success("VGG16 model loaded successfully with TensorFlow!")
            return model
        except Exception as e:
            st.warning(f"TensorFlow loading failed: {str(e)}")
            st.info("Trying with Keras 3...")
            
            # Try with Keras 3
            try:
                import keras
                model = keras.models.load_model(MODEL_PATH)
                st.success("VGG16 model loaded successfully with Keras 3!")
                return model
            except Exception as e2:
                st.error(f"Keras 3 loading also failed: {str(e2)}")
                return None
                
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Attempt to download and verify the model
download_success = download_and_verify_model()

# Load the model
vgg_model = load_vgg_model()

# -------------------------------
# Load Whisper Model
# -------------------------------
@st.cache_resource
def load_whisper_model():
    """Load the Whisper model with error handling"""
    try:
        st.info("Loading Whisper model...")
        import whisper
        model = whisper.load_model("base")
        st.success("Whisper model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

whisper_model = load_whisper_model()

# -------------------------------
# Predict Disease from Image (Fallback)
# -------------------------------
def predict_disease_fallback(image):
    """Simple fallback disease prediction"""
    st.warning("Using fallback prediction method (main model unavailable)")
    
    # Simple heuristic based on color analysis
    image_array = np.array(image.resize((224, 224)))
    avg_green = np.mean(image_array[:, :, 1])  # Green channel
    
    if avg_green > 150:
        return "Healthy", 75.0
    else:
        return "Possible Disease", 65.0

def predict_disease(image):
    """Predict disease from an image with fallback"""
    if vgg_model is None:
        return predict_disease_fallback(image)
    
    class_names = [
        "Apple Scab", "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy",
        "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy",
        "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Northern Leaf Blight", "Corn Healthy",
        "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy",
        "Orange Huanglongbing (Citrus Greening)",
        "Peach Bacterial Spot", "Peach Healthy",
        "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
        "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
        "Raspberry Healthy",
        "Soybean Healthy",
        "Squash Powdery Mildew",
        "Strawberry Leaf Scorch", "Strawberry Healthy",
        "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight",
        "Tomato Leaf Mold", "Tomato Septoria Leaf Spot", "Tomato Spider Mites", 
        "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy"
    ]
    
    try:
        # Check which library we're using
        if hasattr(vgg_model, 'predict'):
            # Standard Keras/TensorFlow model
            image = np.array(image.resize((224, 224))) / 255.0
            image = np.expand_dims(image, axis=0)
            prediction = vgg_model.predict(image)
            
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            return predicted_class, confidence
        else:
            # Fallback if model format is unexpected
            return predict_disease_fallback(image)
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return predict_disease_fallback(image)

# -------------------------------
# AI Chatbot using Ollama
# -------------------------------
def get_chat_response(user_input):
    """Get response from the AI chatbot with error handling"""
    try:
        prompt = f"You are a plant disease expert. Answer based on scientific agricultural knowledge.\nQuestion: {user_input}"
        response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try again later."

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌿 Plant Disease Detection & AI Assistant")

# Model status
if vgg_model is None:
    st.warning("""
    ⚠️ **Plant Disease Model Not Loaded**
    
    The plant disease detection model could not be loaded. This may be due to:
    - Version incompatibility (model was saved with Keras 3.8.0)
    - Corrupted model file download
    - Insufficient memory
    
    Using fallback detection method (limited accuracy).
    """)
    
    # Provide instructions for fixing the issue
    with st.expander("How to fix this issue"):
        st.markdown("""
        ### Solution for Keras 3 Compatibility Issue
        
        Your model was saved with Keras 3.8.0, but this app is trying to load it with TensorFlow's Keras.
        
        **Options:**
        1. **Re-save the model with TensorFlow**: 
           - Load the model in an environment with Keras 3.8.0
           - Save it again using `tf.keras.models.save_model()`
        
        2. **Install Keras 3 in this environment**:
           - Add `keras==3.8.0` to your requirements.txt
           - Modify the code to use `keras.models.load_model()` instead of `tf.keras.models.load_model()`
        
        3. **Use a different model format**:
           - Convert to TensorFlow Lite (.tflite) for better compatibility
           - Use ONNX format for cross-framework compatibility
        """)
else:
    st.success("✅ Plant Disease Model Loaded Successfully!")

if whisper_model is None:
    st.warning("⚠️ Whisper model could not be loaded. Voice features disabled.")
else:
    st.success("✅ Whisper Model Loaded Successfully!")

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["Image Analysis", "Text Chat", "Voice Chat"])

with tab1:
    st.header("📤 Upload Leaf Image for Analysis")
    
    # Upload an Image
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"], key="image_upload")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Predict Disease
        if st.button("Analyze Disease", type="primary"):
            with st.spinner("Analyzing disease..."):
                disease, confidence = predict_disease(image)
                
                st.subheader("🔍 Prediction Results:")
                st.write(f"**Disease Type:** {disease}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                
                # AI Recommendations
                st.subheader("🩺 AI Suggestions:")
                query = f"What is {disease} and how can I treat it?"
                response = get_chat_response(query)
                st.write(response)

with tab2:
    st.header("💬 Text-based Plant Expert Chat")
    
    user_input = st.text_input("Ask a question about plant diseases:", key="text_input")
    if user_input:
        with st.spinner("Thinking..."):
            ai_response = get_chat_response(user_input)
            st.write(f"**AI Answer:** {ai_response}")

with tab3:
    st.header("🎤 Voice-based Plant Expert Chat")
    
    if st.button("Start Voice Recognition", key="voice_button"):
        if whisper_model is None:
            st.error("Whisper model not available. Please try again later.")
        else:
            # Placeholder for voice recognition
            st.info("Voice recognition would be available when the Whisper model is loaded.")

# Add footer with info
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>This plant disease detection app uses a VGG16 model trained on plant leaf images.</p>
    <p>For best results, use clear, well-lit images of plant leaves.</p>
</div>
""", unsafe_allow_html=True)

# Debug information (collapsible)
with st.expander("Debug Information"):
    st.write(f"Model path: {MODEL_PATH}")
    st.write(f"Model exists: {os.path.exists(MODEL_PATH)}")
    if os.path.exists(MODEL_PATH):
        st.write(f"Model size: {os.path.getsize(MODEL_PATH)} bytes")
        st.write(f"Is valid HDF5: {is_valid_h5_file(MODEL_PATH)}")
