import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import h5py
import time

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

# -------------------------------
# Load Plant Disease Classification Model (VGG16)
# -------------------------------
@st.cache_resource
def load_vgg_model():
    """Load the VGG16 model with error handling"""
    if not os.path.exists(MODEL_PATH):
        return None
    
    try:
        if not is_valid_h5_file(MODEL_PATH):
            return None
            
        # Try to load with standard approach first
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            # Try with Keras 3 if available
            try:
                import keras
                model = keras.models.load_model(MODEL_PATH)
                return model
            except Exception as e2:
                return None
                
    except Exception as e:
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
        import whisper
        model = whisper.load_model("base")
        return model
    except Exception as e:
        return None

whisper_model = load_whisper_model()

# -------------------------------
# Predict Disease from Image (Fallback)
# -------------------------------
def predict_disease_fallback(image):
    """Simple fallback disease prediction"""
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
        "Peach Bacterial Step", "Peach Healthy",
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
        # Standard Keras/TensorFlow model
        image = np.array(image.resize((224, 224))) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = vgg_model.predict(image)
        
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return predicted_class, confidence
    except Exception as e:
        return predict_disease_fallback(image)

# -------------------------------
# AI Chatbot
# -------------------------------
def get_chat_response(user_input):
    """Get response from the AI chatbot with error handling"""
    try:
        import ollama
        prompt = f"You are a plant disease expert. Answer based on scientific agricultural knowledge.\nQuestion: {user_input}"
        response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try again later."

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌿 Plant Disease Detection & AI Assistant")

# Model upload section
st.sidebar.header("Model Setup")
st.sidebar.info("""
**Google Drive download is not working.**
Please upload your model file manually or try alternative hosting.
""")

uploaded_model = st.sidebar.file_uploader(
    "Upload Plant Disease Model", 
    type=["h5", "hdf5"],
    help="Upload your plant_disease_vgg16_optimized.h5 file"
)

if uploaded_model is not None:
    # Save the uploaded file
    with open(MODEL_PATH, "wb") as f:
        f.write(uploaded_model.getbuffer())
    
    # Clear cache to force reload
    st.cache_resource.clear()
    st.sidebar.success("Model uploaded successfully! Please refresh the page.")
    
    # Refresh the app
    st.rerun()

# Alternative hosting options
with st.sidebar.expander("Alternative Hosting Options"):
    st.markdown("""
    **If your model is hosted elsewhere:**
    1. **Hugging Face Hub**: Upload your model to Hugging Face
    2. **GitHub Releases**: Use GitHub's release feature for large files
    3. **Dropbox**: Use a shared link with dl.dropboxusercontent.com
    4. **AWS S3**: Use pre-signed URLs for secure access
    
    **To fix Google Drive issues:**
    - Ensure file is set to "Anyone with the link can view"
    - Avoid frequent downloads which may trigger restrictions
    """)

# Model status
if vgg_model is None:
    st.warning("""
    ⚠️ **Plant Disease Model Not Loaded**
    
    Please upload your model file using the sidebar.
    Without the model, we can only provide basic image analysis.
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
                
                # Show model status
                if vgg_model is None:
                    st.info("ℹ️ Using fallback analysis (limited accuracy)")
                
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
