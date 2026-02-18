"""
Plant Disease Recognition Web Application - CORRECT PATHS
"""
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pickle
import os
import time

# Page configuration
st.set_page_config(
    page_title="🌱 Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.8rem; color: #2E8B57; text-align: center; margin-bottom: 1rem; font-weight: 700; }
    .result-box { background-color: #f8f9fa; padding: 25px; border-radius: 15px; margin: 20px 0; border-left: 6px solid #2E8B57; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .disease-info { background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin: 15px 0; }
    .debug-info { background-color: #f0f7ff; padding: 15px; border-radius: 10px; margin: 15px 0; font-family: monospace; font-size: 0.9rem; }
    .warning-box { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 10px; margin: 15px 0; }
</style>
""", unsafe_allow_html=True)

# Load TFLite model - USING YOUR FOLDER STRUCTURE
@st.cache_resource
def load_tflite_model():
    """Load the TensorFlow Lite model"""
    model_path = "models/plant_disease_model.tflite"
    
    if not os.path.exists(model_path):
        st.error(f"""
        ❌ Model file not found at: {model_path}
        
        To create the TFLite model:
        1. Ensure you have trained the model (best_model.h5 exists)
        2. Run: python model_conversion.py
        3. This will create: models/plant_disease_model.tflite
        """)
        return None
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get model info
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        st.sidebar.success(f"✅ Model loaded: models/plant_disease_model.tflite")
        st.sidebar.write(f"**Input:** {input_details[0]['shape']}")
        st.sidebar.write(f"**Output:** {output_details[0]['shape']}")
        st.sidebar.write(f"**Input dtype:** {input_details[0]['dtype']}")
        
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load class names - USING YOUR FOLDER STRUCTURE
@st.cache_resource
def load_class_names():
    """Load class names"""
    # Try root folder first, then models folder
    if os.path.exists("class_names.pkl"):
        class_path = "class_names.pkl"
    elif os.path.exists("models/class_names.txt"):
        # Load from text file and convert to list
        with open("models/class_names.txt", 'r') as f:
            lines = f.readlines()
        class_names = []
        for line in lines[2:]:  # Skip header
            if ':' in line:
                class_name = line.split(':', 1)[1].strip()
                class_names.append(class_name.replace(' - ', '___').replace(' ', '_'))
        st.sidebar.success(f"✅ Loaded {len(class_names)} classes from models/class_names.txt")
        return class_names
    else:
        st.error("""
        ❌ Class names file not found!
        
        Expected: class_names.pkl (root folder) or models/class_names.txt
        
        To create class_names.pkl:
        1. Run: python preprocess_data.py
        2. This will create class_names.pkl in root folder
        """)
        return None
    
    try:
        with open(class_path, 'rb') as f:
            class_names = pickle.load(f)
        
        st.sidebar.success(f"✅ Loaded {len(class_names)} classes from {class_path}")
        
        # Show sample classes in sidebar
        with st.sidebar.expander("View first 5 classes"):
            for i in range(min(5, len(class_names))):
                st.write(f"{i}: {class_names[i]}")
        
        return class_names
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return None

def preprocess_image_correct(image):
    """
    CORRECT preprocessing for your trained model
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    debug_info = {
        'original_shape': img_array.shape,
        'original_dtype': str(img_array.dtype),
    }
    
    # Handle different image formats
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
        debug_info['format'] = 'Grayscale → RGB'
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Remove alpha
        debug_info['format'] = 'RGBA → RGB'
    elif img_array.shape[2] == 3:  # RGB
        debug_info['format'] = 'RGB'
    
    # IMPORTANT: Your training used ImageDataGenerator with PIL
    # PIL loads images as RGB, so keep as RGB
    img_rgb = img_array
    
    # Resize to 128x128 (model input size)
    img_resized = cv2.resize(img_rgb, (128, 128))
    debug_info['resized_shape'] = img_resized.shape
    
    # Normalize to [0, 1] - matches training
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Check normalization
    debug_info['min_val'] = float(img_normalized.min())
    debug_info['max_val'] = float(img_normalized.max())
    debug_info['mean_val'] = float(img_normalized.mean())
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    debug_info['final_shape'] = img_batch.shape
    
    return img_batch, debug_info

def analyze_prediction_bias(predictions, class_names):
    """Check if model is biased toward healthy classes"""
    healthy_indices = []
    healthy_confidences = []
    
    for i, class_name in enumerate(class_names):
        if 'healthy' in class_name.lower():
            healthy_indices.append(i)
            healthy_confidences.append(predictions[i] * 100)
    
    disease_indices = []
    disease_confidences = []
    
    for i, class_name in enumerate(class_names):
        if 'healthy' not in class_name.lower():
            disease_indices.append(i)
            disease_confidences.append(predictions[i] * 100)
    
    avg_healthy = np.mean(healthy_confidences) if healthy_confidences else 0
    avg_disease = np.mean(disease_confidences) if disease_confidences else 0
    
    return {
        'healthy_count': len(healthy_indices),
        'disease_count': len(disease_indices),
        'avg_healthy': avg_healthy,
        'avg_disease': avg_disease,
        'bias_ratio': avg_healthy / avg_disease if avg_disease > 0 else float('inf')
    }

def format_class_name(class_name):
    """Format class name for display"""
    if '___' in class_name:
        plant, disease = class_name.split('___', 1)
        return f"{plant.replace('_', ' ')} - {disease.replace('_', ' ')}"
    return class_name.replace('_', ' ')

def get_disease_info(class_name):
    """Get detailed disease information"""
    class_lower = class_name.lower()
    
    # Apple diseases
    if 'apple' in class_lower:
        if 'black_rot' in class_lower:
            return {
                'name': 'Apple Black Rot',
                'description': 'Fungal disease caused by Botryosphaeria obtusa. Causes dark, sunken lesions on fruit and purple-brown spots on leaves.',
                'symptoms': 'Dark lesions on fruit, purple spots on leaves, cankers on branches',
                'treatment': 'Apply fungicides (captan, thiophanate-methyl). Prune infected branches.',
                'prevention': 'Remove fallen fruit/leaves, prune for air circulation, use resistant varieties',
                'severity': 'High'
            }
        elif 'healthy' in class_lower:
            return {
                'name': 'Healthy Apple',
                'description': 'Apple tree is healthy with no signs of disease.',
                'symptoms': 'None',
                'treatment': 'None required',
                'prevention': 'Regular monitoring, proper fertilization',
                'severity': 'None'
            }
    
    # Corn diseases
    elif 'corn' in class_lower or 'maize' in class_lower:
        if 'healthy' in class_lower:
            return {
                'name': 'Healthy Corn',
                'description': 'Corn plant is healthy with no disease symptoms.',
                'symptoms': 'None',
                'treatment': 'None required',
                'prevention': 'Crop rotation, proper spacing',
                'severity': 'None'
            }
    
    # Default for healthy
    if 'healthy' in class_lower:
        return {
            'name': format_class_name(class_name),
            'description': 'Plant appears healthy with no visible disease symptoms.',
            'symptoms': 'None',
            'treatment': 'Continue regular care',
            'prevention': 'Maintain proper growing conditions',
            'severity': 'None'
        }
    
    # Default for diseases
    return {
        'name': format_class_name(class_name),
        'description': 'Plant disease detected. Specific identification may require expert consultation.',
        'symptoms': 'Varied depending on disease',
        'treatment': 'Consult agricultural expert',
        'prevention': 'Practice good plant hygiene',
        'severity': 'Moderate'
    }

def main():
    # Initialize session state
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    
    # Sidebar
    with st.sidebar:
        st.title("🌿 Plant Disease Detector")
        st.write("""
        **AI Model Information:**
        - Trained on: 38 plant diseases
        - Input size: 128×128 pixels
        - Framework: TensorFlow Lite
        - Accuracy: ~83.5%
        
        **How to use:**
        1. Upload clear leaf image
        2. Click Analyze
        3. View diagnosis
        """)
        
        st.divider()
        
        # Load resources
        st.write("**System Status:**")
        interpreter = load_tflite_model()
        class_names = load_class_names()
        
        if interpreter and class_names:
            st.success("✅ All systems ready")
        
        st.divider()
        
        # File check
        st.write("**Required Files Check:**")
        
        files_to_check = [
            ("models/plant_disease_model.tflite", "TFLite Model"),
            ("class_names.pkl", "Class Names"),
            ("models/best_model.h5", "Keras Model"),
        ]
        
        for file_path, description in files_to_check:
            exists = os.path.exists(file_path)
            icon = "✅" if exists else "❌"
            st.write(f"{icon} {description}: {os.path.basename(file_path)}")
        
        st.divider()
        st.caption("Thesis Project: Plant Disease Recognition")
        st.caption("Files are correctly located in models/ folder")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">🌱 Plant Disease Detector</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666;">Using AI to detect plant diseases from leaf images</p>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "📤 Upload Plant Leaf Image",
            type=['jpg', 'jpeg', 'png'],
            help="Supported: Apple, Tomato, Potato, Corn, Grape, etc."
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            
            # Show original and processed
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                if not interpreter:
                    st.error("Model not loaded. Please check models/plant_disease_model.tflite")
                elif not class_names:
                    st.error("Class names not loaded. Please check class_names.pkl")
                else:
                    with st.spinner("🔬 Analyzing image..."):
                        # Preprocess
                        input_data, debug_info = preprocess_image_correct(image)
                        
                        # Show processed image
                        with col_img2:
                            processed_img = (input_data[0] * 255).astype(np.uint8)
                            st.image(processed_img, caption="Model Input (128×128)", use_column_width=True)
                        
                        # Get model details
                        input_details = interpreter.get_input_details()
                        output_details = interpreter.get_output_details()
                        
                        # Make prediction
                        start_time = time.time()
                        interpreter.set_tensor(input_details[0]['index'], input_data)
                        interpreter.invoke()
                        inference_time = (time.time() - start_time) * 1000
                        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
                        
                        # Analyze bias
                        bias_info = analyze_prediction_bias(predictions, class_names)
                        
                        # Get top prediction
                        top_idx = np.argmax(predictions)
                        top_class = class_names[top_idx]
                        top_conf = predictions[top_idx] * 100
                        
                        # Store in session state
                        st.session_state.last_prediction = {
                            'class': top_class,
                            'confidence': top_conf,
                            'time': inference_time,
                            'bias_ratio': bias_info['bias_ratio']
                        }
                        
                        # Show debug info
                        with st.expander("🔧 Technical Details", expanded=False):
                            st.markdown('<div class="debug-info">', unsafe_allow_html=True)
                            st.write("**Preprocessing:**")
                            for key, value in debug_info.items():
                                st.write(f"- {key}: {value}")
                            
                            st.write(f"\n**Prediction Stats:**")
                            st.write(f"- Total predictions: {len(predictions)}")
                            st.write(f"- Sum: {predictions.sum():.6f}")
                            st.write(f"- Max confidence: {top_conf:.1f}%")
                            st.write(f"- Inference time: {inference_time:.1f} ms")
                            
                            st.write(f"\n**Bias Analysis:**")
                            st.write(f"- Healthy classes: {bias_info['healthy_count']}")
                            st.write(f"- Disease classes: {bias_info['disease_count']}")
                            st.write(f"- Avg healthy confidence: {bias_info['avg_healthy']:.1f}%")
                            st.write(f"- Avg disease confidence: {bias_info['avg_disease']:.1f}%")
                            st.write(f"- Bias ratio: {bias_info['bias_ratio']:.2f}")
                            
                            if bias_info['bias_ratio'] > 2:
                                st.warning("⚠️ Model shows bias toward healthy classes")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display results
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        
                        # Diagnosis with icon
                        display_name = format_class_name(top_class)
                        
                        if 'healthy' in top_class.lower():
                            st.success(f"✅ **Diagnosis: {display_name}**")
                            st.balloons()
                        else:
                            st.error(f"⚠️ **Diagnosis: {display_name}**")
                        
                        # Confidence
                        st.write(f"**Confidence:** {top_conf:.1f}%")
                        st.progress(float(top_conf) / 100)
                        
                        # Disease information
                        disease_info = get_disease_info(top_class)
                        
                        st.markdown('<div class="disease-info">', unsafe_allow_html=True)
                        st.write(f"**📋 {disease_info['name']}**")
                        
                        cols = st.columns(2)
                        with cols[0]:
                            st.write("**Description:**")
                            st.write(disease_info['description'])
                            
                            st.write("**Symptoms:**")
                            st.write(disease_info['symptoms'])
                        
                        with cols[1]:
                            st.write("**Treatment:**")
                            st.write(disease_info['treatment'])
                            
                            st.write("**Prevention:**")
                            st.write(disease_info['prevention'])
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Top 5 predictions
                        with st.expander("📊 All Predictions (Top 10)"):
                            top_10_idx = np.argsort(predictions)[-10:][::-1]
                            
                            for i, idx in enumerate(top_10_idx):
                                conf = predictions[idx] * 100
                                class_display = format_class_name(class_names[idx])
                                
                                # Create a bar chart-like display
                                bar_width = int(conf * 2)  # Scale for visual
                                bar = "█" * (bar_width // 10)  # Every 10% = 1 character
                                
                                if i == 0:
                                    st.write(f"**{i+1}. 🎯 {class_display}**")
                                    st.write(f"   {bar} {conf:.1f}%")
                                else:
                                    st.write(f"{i+1}. {class_display}")
                                    st.write(f"   {bar} {conf:.1f}%")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Show last prediction if exists
            if st.session_state.last_prediction:
                st.divider()
                st.write("**Last Analysis Summary:**")
                last = st.session_state.last_prediction
                st.write(f"- **Class:** {format_class_name(last['class'])}")
                st.write(f"- **Confidence:** {last['confidence']:.1f}%")
                st.write(f"- **Time:** {last['time']:.1f} ms")
                if last['bias_ratio'] > 2:
                    st.write(f"- **Note:** Model bias detected (ratio: {last['bias_ratio']:.1f})")
    
    with col2:
        # Sample guidelines
        st.subheader("📸 Image Guidelines")
        
        st.write("**Good Examples:**")
        good_cols = st.columns(2)
        with good_cols[0]:
            st.image("https://via.placeholder.com/150x150/4CAF50/FFFFFF?text=Clear", 
                    caption="Clear", use_column_width=True)
        with good_cols[1]:
            st.image("https://via.placeholder.com/150x150/2196F3/FFFFFF?text=Focused", 
                    caption="Focused", use_column_width=True)
        
        st.write("**Avoid:**")
        bad_cols = st.columns(2)
        with bad_cols[0]:
            st.image("https://via.placeholder.com/150x150/FF9800/FFFFFF?text=Blurry", 
                    caption="Blurry", use_column_width=True)
        with bad_cols[1]:
            st.image("https://via.placeholder.com/150x150/F44336/FFFFFF?text=Dark", 
                    caption="Dark", use_column_width=True)
        
        # Performance stats
        st.divider()
        st.subheader("📊 Model Performance")
        
        metrics_cols = st.columns(2)
        with metrics_cols[0]:
            st.metric("Accuracy", "83.5%")
            st.metric("Model Size", "0.18 MB")
        
        with metrics_cols[1]:
            st.metric("Speed", "~2.4 ms")
            st.metric("FPS", "416")
        
        # Supported plants
        st.divider()
        st.subheader("🌿 Supported Plants")
        
        plants = [
            "Apple", "Tomato", "Potato", "Corn",
            "Grape", "Peach", "Pepper", "Squash",
            "Soybean", "Strawberry", "Blueberry", "Cherry"
        ]
        
        # Display in 2 columns
        plant_cols = st.columns(2)
        for i, plant in enumerate(plants):
            with plant_cols[i % 2]:
                st.markdown(f'- {plant}')
        
        # Quick tips
        st.divider()
        st.subheader("💡 Tips for Best Results")
        st.write("""
        1. Use natural daylight
        2. Fill frame with leaf
        3. Focus on affected area
        4. Avoid shadows
        5. Take multiple angles
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p><strong>🌍 Plant Disease Recognition System</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the app
if __name__ == "__main__":
    # Quick check for required files
    if not os.path.exists("models/plant_disease_model.tflite"):
        st.error("""
        ⚠️ **Important: TFLite model not found**
        
        Location expected: `models/plant_disease_model.tflite`
        
        **To create it:**
        ```bash
        python model_conversion.py
        ```
        
        This requires:
        1. `models/best_model.h5` (trained model)
        2. `class_names.pkl` (class names)
        """)
    
    if not os.path.exists("class_names.pkl"):
        st.warning("""
        ⚠️ **Class names file not found**
        
        Expected: `class_names.pkl` in root folder
        
        **To create it:**
        ```bash
        python preprocess_data.py
        ```
        """)
    
    main()