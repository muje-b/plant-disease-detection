import streamlit as st
import numpy as np
from PIL import Image
import os
import h5py
import time
import subprocess
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'upload_complete' not in st.session_state:
    st.session_state.upload_complete = False
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="Plant Disease Detection & AI Assistant",
    page_icon="🌿",
    layout="wide"
)

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

def install_keras3():
    """Install Keras 3 to handle the model compatibility"""
    try:
        # Try to import keras 3
        import keras
        if hasattr(keras, '__version__') and keras.__version__.startswith('3'):
            return True
        else:
            # Install keras 3
            subprocess.check_call([sys.executable, "-m", "pip", "install", "keras==3.0.0"])
            return True
    except:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "keras==3.0.0"])
            return True
        except:
            return False

def convert_model_to_tflite(model_path, output_path):
    """Convert a Keras model to TensorFlow Lite format"""
    try:
        # First try to install keras 3 and load the model
        if install_keras3():
            import keras
            model = keras.models.load_model(model_path)
            
            # Convert to TensorFlow Lite
            import tensorflow as tf
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            tflite_model = converter.convert()
            
            # Save the converted model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            return True, "Model converted successfully to TensorFlow Lite format."
        else:
            return False, "Failed to install Keras 3 for conversion."
    except Exception as e:
        return False, f"Conversion failed: {str(e)}"

# -------------------------------
# Model Upload Section
# -------------------------------
def render_model_upload():
    """Render the model upload section with proper state management"""
    st.sidebar.header("Model Configuration")
    
    # Model upload section
    with st.sidebar.expander("Upload Model File", expanded=not st.session_state.model_loaded):
        st.info("Upload your trained plant disease detection model (.h5 format)")
        
        # Use a key that increments on successful upload to reset the uploader
        uploaded_file = st.file_uploader(
            "Choose a .h5 model file", 
            type=["h5"],
            key=f"model_uploader_{st.session_state.file_uploader_key}"
        )
        
        if uploaded_file is not None:
            # Save the uploaded file
            model_path = "uploaded_model.h5"
            with open(model_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Validate the file
            if is_valid_h5_file(model_path):
                # Try to convert the model to TensorFlow Lite format
                with st.spinner("Converting model for compatibility..."):
                    tflite_path = "converted_model.tflite"
                    success, message = convert_model_to_tflite(model_path, tflite_path)
                    
                    if success:
                        st.success(message)
                        
                        # Load the converted model
                        try:
                            import tensorflow as tf
                            interpreter = tf.lite.Interpreter(model_path=tflite_path)
                            interpreter.allocate_tensors()
                            st.session_state.model = interpreter
                            st.session_state.model_loaded = True
                            st.session_state.upload_complete = True
                            st.success("Model converted and loaded successfully!")
                            
                            # Increment the key to reset the uploader
                            st.session_state.file_uploader_key += 1
                            
                            # Add a small delay before rerun to ensure UI updates
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load converted model: {str(e)}")
                    else:
                        st.error(message)
                        st.warning("""
                        **Compatibility Issue Detected**
                        
                        Your model was created with a newer version of Keras (Keras 3.x) which is not 
                        compatible with this environment. 
                        
                        **Recommended Solutions:**
                        1. Re-save your model using TensorFlow's `tf.keras.models.save_model()` function
                        2. Convert your model to TensorFlow Lite format externally
                        3. Use the fallback mode for basic analysis
                        """)
            else:
                st.error("Uploaded file is not a valid HDF5 model file. Please upload a valid Keras model.")

# -------------------------------
# Plant Disease Prediction Functions
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

def predict_disease_tflite(image, interpreter):
    """Predict disease from an image using TensorFlow Lite model"""
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess the image
    image = np.array(image.resize((224, 224))) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    interpreter.invoke()
    
    # Get the prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
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
    
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

def predict_disease(image):
    """Predict disease from an image"""
    if st.session_state.model is None:
        return predict_disease_fallback(image)
    
    try:
        # Check if we're using a TensorFlow Lite model
        if hasattr(st.session_state.model, 'get_input_details'):
            return predict_disease_tflite(image, st.session_state.model)
        else:
            # Standard Keras model prediction (if we had one)
            return predict_disease_fallback(image)
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return predict_disease_fallback(image)

# -------------------------------
# AI Chatbot Function
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
# Main Application UI
# -------------------------------
def main():
    st.title("🌿 Plant Disease Detection & AI Assistant")
    
    # Render the model upload section
    render_model_upload()
    
    # Display model status
    if st.session_state.model_loaded:
        st.sidebar.success("✅ Model Loaded Successfully!")
    else:
        st.sidebar.warning("⚠️ No Model Loaded - Using Fallback Mode")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Image Analysis", "Text Chat", "About"])
    
    with tab1:
        st.header("📤 Plant Disease Detection")
        
        # Image upload section
        st.subheader("Upload Leaf Image")
        image_file = st.file_uploader(
            "Choose a leaf image...", 
            type=["jpg", "png", "jpeg"],
            key="image_uploader"
        )
        
        if image_file is not None:
            # Display the uploaded image
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Analyze button
            if st.button("Analyze Disease", type="primary"):
                with st.spinner("Analyzing..."):
                    # Predict disease
                    disease, confidence = predict_disease(image)
                    
                    # Display results
                    st.subheader("🔍 Analysis Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Condition", disease)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2f}%")
                    
                    # Show model status
                    if not st.session_state.model_loaded:
                        st.info("ℹ️ Using fallback analysis (limited accuracy)")
                    
                    # AI Recommendations
                    st.subheader("🩺 AI Recommendations")
                    with st.spinner("Generating recommendations..."):
                        query = f"What is {disease} in plants and how can I treat it?"
                        response = get_chat_response(query)
                        st.write(response)
    
    with tab2:
        st.header("💬 Plant Expert Chat")
        
        # Text-based chat
        user_input = st.text_input("Ask a question about plant diseases:", key="text_input")
        if user_input:
            with st.spinner("Thinking..."):
                response = get_chat_response(user_input)
                st.write(f"**AI Expert:** {response}")
    
    with tab3:
        st.header("ℹ️ About This App")
        st.markdown("""
        This plant disease detection app uses deep learning to identify diseases in plant leaves.
        
        **Features:**
        - Upload and analyze plant leaf images
        - Get AI-powered disease identification
        - Receive treatment recommendations
        - Chat with a plant disease expert AI
        
        **How to use:**
        1. Upload a trained model file (.h5 format) in the sidebar
        2. Upload a plant leaf image for analysis
        3. View the results and recommendations
        
        **Note:** Without a trained model, the app uses a simple fallback detection method with limited accuracy.
        """)
        
        # Model conversion instructions
        with st.expander("Model Conversion Instructions"):
            st.markdown("""
            ### If your model fails to load:
            
            Your model was likely created with Keras 3.x, which is not compatible with this environment.
            
            **To fix this:**
            
            1. **Re-save your model with TensorFlow:**
               ```python
               # In your training environment
               import tensorflow as tf
               tf.keras.models.save_model(model, "compatible_model.h5")
               ```
            
            2. **Convert to TensorFlow Lite:**
               ```python
               # In your training environment
               converter = tf.lite.TFLiteConverter.from_keras_model(model)
               tflite_model = converter.convert()
               with open('model.tflite', 'wb') as f:
                   f.write(tflite_model)
               ```
            
            3. **Use the converted model in this app**
            """)

# Run the app
if __name__ == "__main__":
    main()
