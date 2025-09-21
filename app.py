import streamlit as st
import numpy as np
from PIL import Image
import os
import time

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'upload_complete' not in st.session_state:
    st.session_state.upload_complete = False

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="Plant Disease Detection & AI Assistant",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------
# Model Loading Functions
# -------------------------------
def load_tflite_model(model_path):
    """Load a TensorFlow Lite model"""
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter, True
    except Exception as e:
        return None, False

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

# -------------------------------
# Model Upload Section
# -------------------------------
def render_model_upload():
    """Render the model upload section"""
    st.sidebar.header("Model Configuration")
    
    # Model upload section
    with st.sidebar.expander("Upload Model File", expanded=not st.session_state.model_loaded):
        st.info("""
        **Important:** Your original model is in Keras 3 format which is not compatible.
        Please upload a TensorFlow Lite (.tflite) model instead.
        """)
        
        # Upload TensorFlow Lite model
        uploaded_file = st.file_uploader(
            "Choose a .tflite model file", 
            type=["tflite"],
            key="model_uploader"
        )
        
        if uploaded_file is not None:
            # Save the uploaded file
            model_path = "uploaded_model.tflite"
            with open(model_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the model
            model, success = load_tflite_model(model_path)
            if success:
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.session_state.upload_complete = True
                st.success("Model loaded successfully!")
                st.rerun()
            else:
                st.error("Failed to load the TensorFlow Lite model.")

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
    
    try:
        # Check if we're using a TensorFlow Lite model
        if hasattr(st.session_state.model, 'get_input_details'):
            return predict_disease_tflite(image, st.session_state.model)
        else:
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
# Conversion Instructions
# -------------------------------
def show_conversion_instructions():
    """Show instructions for converting the model"""
    with st.sidebar.expander("How to Convert Your Model"):
        st.markdown("""
        ### Convert Your Keras 3 Model to TensorFlow Lite
        
        Your model was created with Keras 3.x, which is not compatible with this environment.
        
        **To convert your model:**
        
        1. **Install TensorFlow and Keras 3** in a local environment:
        ```bash
        pip install tensorflow keras
        ```
        
        2. **Run this conversion script:**
        ```python
        import tensorflow as tf
        import keras
        
        # Load your Keras 3 model
        model = keras.models.load_model("plant_disease_vgg16_optimized.h5")
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Save the converted model
        with open('converted_model.tflite', 'wb') as f:
            f.write(tflite_model)
        ```
        
        3. **Upload the converted .tflite file** to this app
        """)

# -------------------------------
# Main Application UI
# -------------------------------
def main():
    st.title("🌿 Plant Disease Detection & AI Assistant")
    
    # Show conversion instructions
    show_conversion_instructions()
    
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
        
        **Note:** For best results, upload a TensorFlow Lite (.tflite) model converted from your Keras 3 model.
        Without a compatible model, the app uses a simple fallback detection method with limited accuracy.
        """)

# Run the app
if __name__ == "__main__":
    main()
