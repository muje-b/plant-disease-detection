import streamlit as st
import tensorflow as tf
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from PIL import Image
import ollama
import os
import gdown
import requests
import h5py
import time
from pathlib import Path

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
    """Check if a file is a valid HDF5 file and contains a Keras model"""
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if it has the basic structure of a Keras model
            if 'model_weights' in f.keys() or 'model_config' in f.keys():
                return True
        return False
    except:
        return False

def download_model_with_retry():
    """Download model with multiple fallback strategies"""
    # Method 1: Try gdown with direct URL
    try:
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
            if is_valid_h5_file(MODEL_PATH):
                return True
            else:
                st.warning("Downloaded file is not a valid HDF5 model")
                os.remove(MODEL_PATH)  # Remove invalid file
    except Exception as e:
        st.warning(f"gdown failed: {e}")
    
    # Method 2: Try alternative Google Drive download URL
    try:
        url = f"https://docs.google.com/uc?export=download&id={FILE_ID}"
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Handle Google Drive virus scan warning
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                url += f"&confirm={value}"
                response = session.get(url, stream=True)
                break
        
        # Download the file with progress
        file_size = int(response.headers.get('Content-Length', 0))
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with open(MODEL_PATH, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if file_size > 0:
                        progress = min(downloaded / file_size, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded {downloaded}/{file_size} bytes ({progress*100:.1f}%)")
        
        progress_bar.empty()
        status_text.empty()
        
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
            if is_valid_h5_file(MODEL_PATH):
                return True
            else:
                st.error("Downloaded file is not a valid HDF5 model")
                return False
    except Exception as e:
        st.error(f"Alternative download failed: {e}")
    
    return False

# -------------------------------
# Download and Verify Model
# -------------------------------
@st.cache_resource
def download_and_verify_model():
    """Download the model with verification"""
    # Check if file already exists and is valid
    if os.path.exists(MODEL_PATH):
        if is_valid_h5_file(MODEL_PATH):
            st.success("Model already exists and is valid!")
            return True
        else:
            st.warning("Model file exists but is corrupted. Redownloading...")
            os.remove(MODEL_PATH)
    
    # Download the model
    with st.spinner("Downloading VGG16 model from Google Drive..."):
        if download_model_with_retry():
            st.success("Model downloaded and verified successfully!")
            return True
        else:
            st.error("Failed to download a valid model file.")
            return False

# Attempt to download and verify the model
download_success = download_and_verify_model()

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
            model = tf.keras.models.load_model(MODEL_PATH)
            st.success("VGG16 model loaded successfully with TensorFlow!")
            return model
        except Exception as e:
            st.warning(f"TensorFlow loading failed: {str(e)}")
            st.info("Trying with Keras 3...")
            
            # Try with Keras 3 if available
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
# Speech Recognition using Whisper
# -------------------------------
def recognize_speech():
    """Recognize speech with error handling"""
    if whisper_model is None:
        st.error("Whisper model not available. Speech recognition disabled.")
        return None
        
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening...")
            audio = recognizer.listen(source, timeout=5)

        audio_data = audio.get_wav_data()
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data)

        result = whisper_model.transcribe("temp_audio.wav")
        user_text = result["text"]
        
        if not user_text.strip():
            st.error("❌ No speech detected. Please try again.")
            return None

        st.success(f"🗣️ You said: {user_text}")
        return user_text
    except sr.WaitTimeoutError:
        st.error("❌ Listening timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error in speech recognition: {str(e)}")
        return None

# -------------------------------
# Text-to-Speech (TTS) using gTTS
# -------------------------------
def text_to_speech(response_text):
    """Convert text to speech with error handling"""
    try:
        tts = gTTS(text=response_text, lang="en")
        audio_path = "response.mp3"
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

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
        ### Solution for Model Loading Issues
        
        1. **Check Google Drive Permissions**:
           - Ensure the model file is set to "Anyone with the link can view" in Google Drive
        
        2. **Re-download the Model**:
           - Click the button below to attempt to download the model again
           - If this fails, consider hosting the model on an alternative platform
        
        3. **Model Compatibility**:
           - Your model was saved with Keras 3.8.0 but this app uses TensorFlow's Keras
           - Consider re-saving the model with TensorFlow format
        """)
        
        if st.button("Retry Model Download"):
            if download_and_verify_model():
                st.rerun()
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
            voice_text = recognize_speech()
            if voice_text:
                with st.spinner("Processing your question..."):
                    ai_response = get_chat_response(voice_text)
                    st.write(f"**Your question:** {voice_text}")
                    st.write(f"**AI Answer:** {ai_response}")
                    
                    # Convert AI Response to Speech and Play in UI
                    audio_file_path = text_to_speech(ai_response)
                    if audio_file_path:
                        with open(audio_file_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format="audio/mp3")

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
        
    # Add a button to manually retry download
    if st.button("Manual Download Retry"):
        if download_and_verify_model():
            st.rerun()
