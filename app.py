import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import atexit
import os

# Initialize session state with proper default values
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'file_uploader_counter' not in st.session_state:
    st.session_state.file_uploader_counter = 0
if 'image_uploader_counter' not in st.session_state:
    st.session_state.image_uploader_counter = 0
if 'model_validated' not in st.session_state:
    st.session_state.model_validated = False
if 'last_model_path' not in st.session_state:
    st.session_state.last_model_path = ""
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []

# Cleanup function for temporary files
def cleanup_temp_files():
    """Clean up temporary files"""
    for temp_file in st.session_state.temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass  # Ignore cleanup errors
    st.session_state.temp_files = []

# Register cleanup function
atexit.register(cleanup_temp_files)

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
    """Load a TensorFlow Lite model with validation"""
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Validate the model is functional
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test with a sample input
        test_input = np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        
        return interpreter, True
    except ImportError:
        st.error("TensorFlow is not installed. Please install TensorFlow to use .tflite models.")
        return None, False
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, False

def validate_model_state():
    """Validate that the model is properly loaded and functional"""
    if st.session_state.model_loaded and st.session_state.model is not None:
        try:
            # Test the model with sample data
            if hasattr(st.session_state.model, 'get_input_details'):
                interpreter = st.session_state.model
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Create test input
                test_input = np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
                
                st.session_state.model_validated = True
                return True
        except Exception as e:
            st.error(f"Model validation failed: {str(e)}")
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.session_state.model_validated = False
            return False
    return False

# -------------------------------
# Plant Disease Prediction Functions
# -------------------------------
def predict_disease_fallback(image):
    """Simple fallback disease prediction"""
    # Simple heuristic based on color analysis
    image_array = np.array(image.resize((224, 224)))
    
    # Check if image has color channels
    if len(image_array.shape) == 3 and image_array.shape[2] >= 3:
        avg_green = np.mean(image_array[:, :, 1])  # Green channel
    else:
        # Grayscale image - use overall brightness
        avg_green = np.mean(image_array)
    
    if avg_green > 150:
        return "Healthy", 75.0
    else:
        return "Possible Disease", 65.0

def predict_disease_tflite(image, interpreter):
    """Predict disease from an image using TensorFlow Lite model"""
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get model's expected input shape
    input_shape = input_details[0]['shape']
    target_height, target_width = input_shape[1], input_shape[2]
    
    # Preprocess the image according to model requirements
    image = np.array(image.resize((target_width, target_height))) / 255.0
    
    # Ensure proper input format
    if len(input_shape) == 4:  # Batch dimension present
        image = np.expand_dims(image, axis=0).astype(np.float32)
    else:
        image = image.astype(np.float32)
    
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
    
    # Safe indexing with bounds checking
    predicted_class_idx = np.argmax(prediction)
    if predicted_class_idx < len(class_names):
        predicted_class = class_names[predicted_class_idx]
    else:
        predicted_class = f"Unknown Class {predicted_class_idx}"
    
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

def predict_disease(image):
    """Predict disease from an image"""
    if st.session_state.model is None or not st.session_state.model_loaded:
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
    except ImportError:
        return "AI chat feature requires Ollama to be installed. Please install Ollama and the 'tinyllama' model to use this feature."
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try again later."

# -------------------------------
# Image Upload Functions
# -------------------------------
def render_image_uploader():
    """Render the image upload section with proper error handling"""
    st.header("📤 Plant Disease Detection")
    
    # Create a unique key for the file uploader
    uploader_key = f"image_uploader_{st.session_state.image_uploader_counter}"
    
    st.subheader("Upload Leaf Image")
    image_file = st.file_uploader(
        "Choose a leaf image...", 
        type=["jpg", "png", "jpeg"],
        key=uploader_key
    )
    
    if image_file is not None:
        try:
            # Validate the uploaded file
            if image_file.size == 0:
                st.error("Uploaded file is empty. Please try again.")
                return None
                
            # Check file type
            if image_file.type not in ["image/jpeg", "image/png", "image/jpg"]:
                st.error("Unsupported file format. Please upload a JPG or PNG image.")
                return None
                
            # Open and validate the image
            image = Image.open(image_file)
            image.verify()  # Verify that it's a valid image file
            
            # Reset the image pointer after verify()
            image = Image.open(image_file)
            
            # Save the image to a temporary file to avoid BytesIO issues
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file.name)
                st.session_state.temp_image_path = tmp_file.name
                # Track the temporary file for cleanup
                st.session_state.temp_files.append(tmp_file.name)
            
            return image
            
        except Exception as e:
            st.error(f"Error processing uploaded image: {str(e)}")
            # Increment the uploader counter to reset the uploader
            st.session_state.image_uploader_counter += 1
            st.rerun()
    
    return None

def process_uploaded_image(image):
    """Process uploaded image for disease prediction"""
    try:
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Revalidate model state before prediction
        if not validate_model_state():
            st.session_state.model_loaded = False
            st.session_state.model = None
        
        # Analyze button
        if st.button("Analyze Disease", type="primary", key="analyze_button"):
            with st.spinner("Analyzing..."):
                # Predict disease
                disease, confidence = predict_disease(image)
                
                # Display results
                st.subheader("🔍 Analysis Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Condition", disease)
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}%")
                with col3:
                    model_status = "TFLite Model" if st.session_state.model_loaded else "Fallback"
                    st.metric("Model Used", model_status)
                
                # Show model status
                if not st.session_state.model_loaded:
                    st.info("ℹ️ Using fallback analysis (limited accuracy)")
                
                # AI Recommendations
                st.subheader("🩺 AI Recommendations")
                with st.spinner("Generating recommendations..."):
                    query = f"What is {disease} in plants and how can I treat it?"
                    response = get_chat_response(query)
                    st.write(response)
                    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        # Reset the file uploader
        st.session_state.image_uploader_counter += 1
        st.rerun()

# -------------------------------
# Model Upload Section
# -------------------------------
def render_model_upload():
    """Render the model upload section"""
    st.sidebar.header("Model Configuration")
    
    # Model upload section
    uploaded_file = st.sidebar.file_uploader(
        "Choose a .tflite model file", 
        type=["tflite"],
        help="Upload your model in TensorFlow Lite format",
        key=f"model_uploader_{st.session_state.file_uploader_counter}"
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
            st.session_state.model_validated = True
            st.session_state.last_model_path = model_path
            st.sidebar.success("Model loaded and validated successfully!")
            # Increment the counter to reset the uploader
            st.session_state.file_uploader_counter += 1
            st.rerun()
        else:
            st.sidebar.error("Failed to load the TensorFlow Lite model.")
            # Increment the counter to reset the uploader
            st.session_state.file_uploader_counter += 1

# -------------------------------
# Main Application UI
# -------------------------------
def main():
    st.title("🌿 Plant Disease Detection & AI Assistant")
    
    # Render the model upload section
    render_model_upload()
    
    # Validate model state and display appropriate status
    if validate_model_state():
        st.sidebar.success("✅ Model Loaded Successfully!")
    else:
        st.sidebar.warning("⚠️ No Model Loaded - Using Fallback Mode")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Image Analysis", "Text Chat", "About"])
    
    with tab1:
        # Render image uploader
        uploaded_image = render_image_uploader()
        
        if uploaded_image is not None:
            process_uploaded_image(uploaded_image)
    
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
        
        **Note:** For best results, convert your model to TensorFlow Lite format and upload it.
        Without a compatible model, the app uses a simple fallback detection method with limited accuracy.
        """)

# Run the app
if __name__ == "__main__":
    main()
