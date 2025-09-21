import streamlit as st
import tensorflow as tf
import numpy as np
import torch
import speech_recognition as sr
import whisper
from gtts import gTTS
from PIL import Image
import ollama
import io
import os
import gdown
import time
import requests
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
# Download VGG16 model from Google Drive with error handling
# -------------------------------
MODEL_PATH = "plant_disease_vgg16_optimized.h5"
FILE_ID = "1IYZz7ibvzsngy0mFbsUFFFLxjbT7_0Bi"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def download_model():
    """Download the model with retries and progress indicators"""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        st.info("Model already downloaded.")
        return True
    
    with st.spinner("Downloading VGG16 model from Google Drive (this may take a few minutes)..."):
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Download with gdown
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            
            # Simulate progress updates since gdown doesn't provide callbacks
            for i in range(100):
                time.sleep(0.05)  # Simulate download time
                progress_bar.progress(i + 1)
                status_text.text(f"Downloading... {i+1}%")
            
            progress_bar.empty()
            status_text.empty()
            
            if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
                st.success("Model downloaded successfully!")
                return True
            else:
                st.error("Download failed - file is empty or doesn't exist")
                return False
                
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
            return False

# Attempt to download the model
download_success = download_model()

# -------------------------------
# Load Plant Disease Classification Model (VGG16)
# -------------------------------
@st.cache_resource
def load_vgg_model():
    """Load the VGG16 model with error handling"""
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please check the download.")
        return None
    
    try:
        with st.spinner("Loading VGG16 model..."):
            model = tf.keras.models.load_model(MODEL_PATH)
            st.success("VGG16 model loaded successfully!")
            return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

vgg_model = load_vgg_model()

# -------------------------------
# Load Whisper Model
# -------------------------------
@st.cache_resource
def load_whisper_model():
    """Load the Whisper model with error handling"""
    try:
        with st.spinner("Loading Whisper model..."):
            model = whisper.load_model("base")
            st.success("Whisper model loaded successfully!")
            return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

whisper_model = load_whisper_model()

# -------------------------------
# Predict Disease from Image
# -------------------------------
def predict_disease(image):
    """Predict disease from an image with error handling"""
    if vgg_model is None:
        st.error("VGG16 model not available. Cannot make predictions.")
        return "Model not available", 0
    
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
        image = np.array(image.resize((224, 224))) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = vgg_model.predict(image)
        
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return "Prediction error", 0

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

# Show warning if model isn't loaded
if vgg_model is None:
    st.warning("""
    ⚠️ **Model Not Loaded**
    
    The plant disease detection model could not be loaded. This may be due to:
    - Large file download issues from Google Drive
    - Insufficient memory on Streamlit Cloud
    - Model compatibility issues
    
    Please try refreshing the app or check the logs for more details.
    """)

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
                
                if disease != "Prediction error":
                    st.subheader("🔍 Prediction Results:")
                    st.write(f"**Disease Type:** {disease}")
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    
                    # AI Recommendations
                    st.subheader("🩺 AI Suggestions:")
                    query = f"What is {disease} and how can I treat it?"
                    response = get_chat_response(query)
                    st.write(response)
                else:
                    st.error("Failed to analyze the image. Please try again.")

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

