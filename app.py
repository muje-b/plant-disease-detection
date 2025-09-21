import streamlit as st
import numpy as np
from PIL import Image
import os
import h5py
import time

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None  # 'h5' or 'tflite'

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
        return interpreter, True, "tflite"
    except Exception as e:
        return None, False, f"Error loading TensorFlow Lite model: {str(e)}"

def load_h5_model(model_path):
    """Load a Keras .h5 model with compatibility handling"""
    try:
        # First try with standard TensorFlow loading
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model, True, "h5"
    except Exception as e:
        error_msg = f"Standard loading failed: {str(e)}"
        
        # Try with custom objects for Keras 3 compatibility
        try:
            custom_objects = {}
            
            # Try to handle DTypePolicy issue
            try:
                from keras import DTypePolicy
                custom_objects['DTypePolicy'] = DTypePolicy
            except:
                pass
                
            # Try to handle other potential custom objects
            try:
                from keras.src.optimizers import Adam
                custom_objects['Adam'] = Adam
            except:
                pass
                
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            return model, True, "h5"
        except Exception as e2:
            return None, False, f"{error_msg}\nCustom objects loading also failed: {str(e2)}"

def is_valid_h5_file(file_path):
    """Check if a file is a valid HDF5 file"""
    try:
        with h5py.File(file_path, 'r') as f:
            return True
    except:
        return False

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

def predict_disease_h5(image, model):
    """Predict disease from an image using Keras model"""
    # Preprocess the image
    image = np.array(image.resize((224, 224))) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    prediction = model.predict(image)
    
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
        if st.session_state.model_type == "tflite":
            return predict_disease_tflite(image, st.session_state.model)
        elif st.session_state.model_type == "h5":
            return predict_disease_h5(image, st.session_state.model)
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
# Model Upload Section
# -------------------------------
def render_model_upload():
    """Render the model upload section"""
    st.sidebar.header("Model Configuration")
    
    # Model upload section
    uploaded_file = st.sidebar.file_uploader(
        "Choose a model file", 
        type=["h5", "tflite"],
        help="Upload your trained model in .h5 or .tflite format"
    )
    
    if uploaded_file is not None:
        # Determine file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Save the uploaded file
        model_path = f"uploaded_model.{file_extension}"
        with open(model_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the model based on file type
        if file_extension == "tflite":
            model, success, message = load_tflite_model(model_path)
        elif file_extension == "h5":
            # Validate the file first
            if not is_valid_h5_file(model_path):
                st.sidebar.error("Uploaded file is not a valid HDF5 file.")
                return
            model, success, message = load_h5_model(model_path)
        else:
            st.sidebar.error(f"Unsupported file format: {file_extension}")
            return
        
        if success:
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.session_state.model_type = file_extension
            st.sidebar.success("Model loaded successfully!")
            st.rerun()
        else:
            st.sidebar.error(f"Failed to load model: {message}")
            
            # Show conversion options for .h5 files
            if file_extension == "h5":
                with st.sidebar.expander("Conversion Options"):
                    st.markdown("""
                    Your .h5 model might be incompatible with this environment.
                    
                    **Options:**
                    1. Try to convert your model to TensorFlow Lite format
                    2. Use the fallback mode for basic analysis
                    
                    **To convert your model:**
                    ```python
                    import tensorflow as tf
                    
                    # Load your model
                    model = tf.keras.models.load_model("your_model.h5")
                    
                    # Convert to TensorFlow Lite
                    converter = tf.lite.TFLiteConverter.from_keras_model(model)
                    tflite_model = converter.convert()
                    
                    # Save the converted model
                    with open('converted_model.tflite', 'wb') as f:
                        f.write(tflite_model)
                    ```
                    """)

# -------------------------------
# Main Application UI
# -------------------------------
def main():
    st.title("🌿 Plant Disease Detection & AI Assistant")
    
    # Render the model upload section
    render_model_upload()
    
    # Display model status
    if st.session_state.model_loaded:
        st.sidebar.success(f"✅ {st.session_state.model_type.upper()} Model Loaded Successfully!")
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
        
        **Supported Model Formats:**
        - Keras .h5 models
        - TensorFlow Lite .tflite models
        
        **Note:** If your .h5 model fails to load, it might be incompatible with this environment.
        Consider converting it to TensorFlow Lite format for better compatibility.
        """)

# Run the app
if __name__ == "__main__":
    main()
