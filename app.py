import streamlit as st
import numpy as np
from PIL import Image
import os
import h5py
import time
import json

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

def fix_model_config(model_path):
    """Attempt to fix model configuration for compatibility"""
    try:
        # Read the model configuration
        with h5py.File(model_path, 'r+') as f:
            if 'model_config' in f.attrs:
                model_config = json.loads(f.attrs['model_config'])
                
                # Fix InputLayer configuration
                if 'config' in model_config and 'layers' in model_config['config']:
                    for layer in model_config['config']['layers']:
                        if layer['class_name'] == 'InputLayer' and 'batch_shape' in layer['config']:
                            # Replace batch_shape with batch_input_shape
                            layer['config']['batch_input_shape'] = layer['config']['batch_shape']
                            del layer['config']['batch_shape']
                    
                    # Write the fixed configuration back
                    f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
                
                return True
    except Exception as e:
        st.error(f"Error fixing model config: {str(e)}")
        return False

def load_model_with_compatibility(model_path):
    """Load model with compatibility fixes"""
    try:
        # First try standard loading
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model, True
    except Exception as e:
        st.warning(f"Standard loading failed: {str(e)}")
        
        # Try with compatibility fixes
        try:
            # Fix the model configuration
            if fix_model_config(model_path):
                # Try loading again
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                return model, True
        except Exception as e2:
            st.error(f"Compatibility loading also failed: {str(e2)}")
            return None, False

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
                # Load the model with compatibility fixes
                model, success = load_model_with_compatibility(model_path)
                if success:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.session_state.upload_complete = True
                    st.success("Model uploaded and loaded successfully!")
                    
                    # Increment the key to reset the uploader
                    st.session_state.file_uploader_key += 1
                    
                    # Add a small delay before rerun to ensure UI updates
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Failed to load the model. Please try a different file.")
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

def predict_disease(image):
    """Predict disease from an image"""
    if st.session_state.model is None:
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
        # Preprocess the image
        image = np.array(image.resize((224, 224))) / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = st.session_state.model.predict(image)
        
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return predicted_class, confidence
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
        
        # Debug information
        with st.expander("Debug Information"):
            st.write(f"Model loaded: {st.session_state.model_loaded}")
            if st.session_state.model_loaded:
                try:
                    st.write("Model architecture:")
                    # This might not work for all models, so we wrap it in a try-except
                    summary_list = []
                    st.session_state.model.summary(print_fn=lambda x: summary_list.append(x))
                    st.text("\n".join(summary_list))
                except:
                    st.write("Model summary not available")

# Run the app
if __name__ == "__main__":
    main()
