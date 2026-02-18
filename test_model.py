"""Test model with specific images"""
import numpy as np
import cv2
import tensorflow as tf
import pickle
import os
from pathlib import Path

# Load model
print("Loading TFLite model...")
try:
    interpreter = tf.lite.Interpreter(model_path="models/plant_disease_model.tflite")
    interpreter.allocate_tensors()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# Load class names
try:
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    print(f"✓ Loaded {len(class_names)} class names")
except Exception as e:
    print(f"✗ Error loading class names: {e}")
    exit(1)

def test_image(image_path):
    """Test a single image"""
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"✗ File not found: {image_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"{'='*60}")
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"✗ Could not read image: {image_path}")
            return None
        
        print(f"✓ Image loaded: {img.shape}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess (must match training preprocessing)
        img_resized = cv2.resize(img, (128, 128))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0).astype(np.float32)
        
        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Get top 10 predictions
        top_10_idx = np.argsort(predictions)[-10:][::-1]
        
        print(f"\nTop Predictions:")
        print("-" * 60)
        for i, idx in enumerate(top_10_idx):
            class_name = class_names[idx]
            conf = predictions[idx] * 100
            formatted_name = class_name.replace('___', ' - ').replace('_', ' ')
            print(f"{i+1:2d}. {formatted_name:45s}: {conf:6.1f}%")
        
        # Show tomato-specific predictions
        print(f"\nTomato-related predictions:")
        print("-" * 60)
        tomato_indices = [i for i, name in enumerate(class_names) 
                         if 'tomato' in name.lower()]
        for idx in tomato_indices:
            conf = predictions[idx] * 100
            if conf > 0.1:  # Show if >0.1%
                formatted_name = class_names[idx].replace('___', ' - ').replace('_', ' ')
                print(f"  {formatted_name:45s}: {conf:6.1f}%")
        
        # Show grape-related predictions
        print(f"\nGrape-related predictions:")
        print("-" * 60)
        grape_indices = [i for i, name in enumerate(class_names) 
                        if 'grape' in name.lower()]
        for idx in grape_indices:
            conf = predictions[idx] * 100
            if conf > 0.1:  # Show if >0.1%
                formatted_name = class_names[idx].replace('___', ' - ').replace('_', ' ')
                print(f"  {formatted_name:45s}: {conf:6.1f}%")
        
        return predictions
        
    except Exception as e:
        print(f"✗ Error processing image: {e}")
        return None

def list_test_images():
    """List available test images"""
    test_dir = "test_images"
    if os.path.exists(test_dir):
        images = [f for f in os.listdir(test_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\nFound {len(images)} test images in '{test_dir}' folder:")
        for img in images[:10]:  # Show first 10
            print(f"  - {img}")
        if len(images) > 10:
            print(f"  ... and {len(images) - 10} more")
        return [os.path.join(test_dir, img) for img in images]
    else:
        print(f"\n'{test_dir}' folder not found. Create it and add test images.")
        return []

def main():
    print("="*60)
    print("Plant Disease Model Tester")
    print("="*60)
    
    # Test with individual image
    test_images = [
        r"C:\Users\govin\Downloads\hgic-veg-septoria-leaf-spot-600.jpg",  # Raw string
        r"D:\Plant_Disease_Recognition\test_tomato.jpg",  # Try local
        "test_tomato.jpg"
    ]
    
    # Find first existing image
    image_to_test = None
    for img_path in test_images:
        if os.path.exists(img_path):
            image_to_test = img_path
            break
    
    if image_to_test:
        predictions = test_image(image_to_test)
        
        if predictions is not None:
            # Create visualization
            visualize_predictions(predictions, image_to_test)
    else:
        print("\nNo test image found. Please:")
        print("1. Copy your tomato image to: D:\\Plant_Disease_Recognition\\test_tomato.jpg")
        print("2. Or create a 'test_images' folder and add images there")
        print("3. Or specify the full path to your image")
        
        # Try to list available images
        available_images = list_test_images()
        if available_images:
            print("\nTesting available images:")
            for img_path in available_images[:3]:  # Test first 3
                test_image(img_path)

def visualize_predictions(predictions, image_path):
    """Create visualization of predictions"""
    import matplotlib.pyplot as plt
    
    # Load and display image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get top 5 predictions
    top_5_idx = np.argsort(predictions)[-5:][::-1]
    top_5_names = [class_names[i] for i in top_5_idx]
    top_5_conf = [predictions[i] * 100 for i in top_5_idx]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show image
    ax1.imshow(img)
    ax1.set_title("Test Image", fontsize=14)
    ax1.axis('off')
    
    # Show predictions as horizontal bars
    y_pos = np.arange(len(top_5_names))
    colors = ['#2E8B57', '#3CB371', '#90EE90', '#C1E1C1', '#E8F5E9']
    
    formatted_names = []
    for name in top_5_names:
        formatted = name.replace('___', ' - ').replace('_', ' ')
        if len(formatted) > 40:
            formatted = formatted[:37] + "..."
        formatted_names.append(formatted)
    
    ax2.barh(y_pos, top_5_conf, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(formatted_names, fontsize=11)
    ax2.invert_yaxis()  # Highest confidence on top
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Top 5 Predictions', fontsize=14)
    
    # Add confidence values on bars
    for i, conf in enumerate(top_5_conf):
        ax2.text(conf + 1, i, f'{conf:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved as 'test_results.png'")
    plt.show()

if __name__ == "__main__":
    main()