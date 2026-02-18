"""
Model Verification Script
Tests Keras model vs TFLite model predictions and helps diagnose issues
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def verify_model_loading():
    """Verify both Keras and TFLite models load correctly"""
    print("="*70)
    print("MODEL LOADING VERIFICATION")
    print("="*70)
    
    # Check if model files exist
    keras_model_path = "models/final_model.h5"
    tflite_model_path = "models/plant_disease_model.tflite"
    
    print(f"Keras model: {keras_model_path} - {'✅ Found' if os.path.exists(keras_model_path) else '❌ Not found'}")
    print(f"TFLite model: {tflite_model_path} - {'✅ Found' if os.path.exists(tflite_model_path) else '❌ Not found'}")
    
    # Load models
    keras_model = None
    tflite_interpreter = None
    
    if os.path.exists(keras_model_path):
        try:
            keras_model = keras.models.load_model(keras_model_path)
            print(f"✅ Keras model loaded successfully")
            print(f"   Input shape: {keras_model.input_shape}")
            print(f"   Output shape: {keras_model.output_shape}")
            print(f"   Parameters: {keras_model.count_params():,}")
        except Exception as e:
            print(f"❌ Failed to load Keras model: {e}")
    
    if os.path.exists(tflite_model_path):
        try:
            tflite_interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            tflite_interpreter.allocate_tensors()
            print(f"✅ TFLite model loaded successfully")
            
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()
            
            print(f"   Input shape: {input_details[0]['shape']}")
            print(f"   Output shape: {output_details[0]['shape']}")
            print(f"   Input dtype: {input_details[0]['dtype']}")
        except Exception as e:
            print(f"❌ Failed to load TFLite model: {e}")
    
    return keras_model, tflite_interpreter

def verify_class_names():
    """Verify class names are loaded correctly"""
    print("\n" + "="*70)
    print("CLASS NAMES VERIFICATION")
    print("="*70)
    
    class_names = []
    
    # Try to load from pickle
    if os.path.exists('class_names.pkl'):
        try:
            with open('class_names.pkl', 'rb') as f:
                class_names = pickle.load(f)
            print(f"✅ Loaded {len(class_names)} classes from class_names.pkl")
        except Exception as e:
            print(f"❌ Failed to load class_names.pkl: {e}")
    
    # Try to load from text file
    elif os.path.exists('models/class_names.txt'):
        try:
            with open('models/class_names.txt', 'r') as f:
                lines = f.readlines()
            class_names = []
            for line in lines[2:]:  # Skip header lines
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        class_name = parts[1].strip()
                        class_names.append(class_name.replace(' - ', '___').replace(' ', '_'))
            print(f"✅ Loaded {len(class_names)} classes from class_names.txt")
        except Exception as e:
            print(f"❌ Failed to load class_names.txt: {e}")
    
    else:
        print("⚠️ No class names file found. Using default names.")
        class_names = [f"Class_{i}" for i in range(38)]
    
    print(f"\nFirst 5 classes:")
    for i in range(min(5, len(class_names))):
        print(f"  {i}: {class_names[i]}")
    
    print(f"\nLast 5 classes:")
    for i in range(max(0, len(class_names)-5), len(class_names)):
        print(f"  {i}: {class_names[i]}")
    
    return class_names

def create_test_inputs():
    """Create various test inputs for verification"""
    print("\n" + "="*70)
    print("TEST INPUTS CREATION")
    print("="*70)
    
    # Test 1: Uniform gray image
    gray_image = np.ones((1, 128, 128, 3), dtype=np.float32) * 0.5
    
    # Test 2: Random noise
    random_image = np.random.randn(1, 128, 128, 3).astype(np.float32) * 0.1 + 0.5
    random_image = np.clip(random_image, 0, 1)
    
    # Test 3: Gradient image
    gradient = np.linspace(0, 1, 128).reshape(1, 128, 1, 1)
    gradient_image = np.repeat(gradient, 128, axis=2)
    gradient_image = np.repeat(gradient_image, 3, axis=3).astype(np.float32)
    
    # Test 4: Checkerboard pattern
    checkerboard = np.zeros((1, 128, 128, 3), dtype=np.float32)
    for i in range(128):
        for j in range(128):
            if (i // 16 + j // 16) % 2 == 0:
                checkerboard[0, i, j, :] = 0.8
            else:
                checkerboard[0, i, j, :] = 0.2
    
    test_inputs = {
        'gray': gray_image,
        'random': random_image,
        'gradient': gradient_image,
        'checkerboard': checkerboard
    }
    
    print(f"Created {len(test_inputs)} test inputs")
    for name, arr in test_inputs.items():
        print(f"  {name}: shape={arr.shape}, range=[{arr.min():.3f}, {arr.max():.3f}]")
    
    return test_inputs

def compare_predictions(keras_model, tflite_interpreter, test_inputs, class_names):
    """Compare predictions between Keras and TFLite models"""
    print("\n" + "="*70)
    print("PREDICTION COMPARISON")
    print("="*70)
    
    if keras_model is None:
        print("❌ Keras model not available for comparison")
        return
    
    if tflite_interpreter is None:
        print("❌ TFLite model not available for comparison")
        return
    
    # Get TFLite details
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    
    results = {}
    
    for test_name, test_input in test_inputs.items():
        print(f"\nTesting: {test_name}")
        print("-" * 40)
        
        # Keras prediction
        keras_pred = keras_model.predict(test_input, verbose=0)[0]
        keras_top = np.argmax(keras_pred)
        keras_conf = keras_pred[keras_top] * 100
        
        # TFLite prediction
        tflite_interpreter.set_tensor(input_details[0]['index'], test_input)
        tflite_interpreter.invoke()
        tflite_pred = tflite_interpreter.get_tensor(output_details[0]['index'])[0]
        tflite_top = np.argmax(tflite_pred)
        tflite_conf = tflite_pred[tflite_top] * 100
        
        # Compare
        match = keras_top == tflite_top
        confidence_diff = abs(keras_conf - tflite_conf)
        
        print(f"  Keras:   {class_names[keras_top]} ({keras_conf:.1f}%)")
        print(f"  TFLite:  {class_names[tflite_top]} ({tflite_conf:.1f}%)")
        print(f"  Match:   {'✅' if match else '❌'}")
        print(f"  Confidence diff: {confidence_diff:.2f}%")
        
        # Store results
        results[test_name] = {
            'keras_top': keras_top,
            'keras_conf': keras_conf,
            'tflite_top': tflite_top,
            'tflite_conf': tflite_conf,
            'match': match,
            'confidence_diff': confidence_diff
        }
    
    return results

def test_with_real_image(image_path, keras_model, tflite_interpreter, class_names):
    """Test with a real image from the dataset"""
    print("\n" + "="*70)
    print("REAL IMAGE TEST")
    print("="*70)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    try:
        # Load and preprocess image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        print(f"Original image: {img_array.shape}")
        
        # Preprocess (same as web app)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Convert RGB to BGR (for OpenCV-style processing)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Resize
        img_resized = cv2.resize(img_bgr, (128, 128))
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        print(f"Processed image: {img_batch.shape}")
        print(f"Value range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")
        
        # Display image
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        plt.title('Resized (128x128)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('test_image_preview.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Get TFLite details
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        
        # Keras prediction
        keras_pred = keras_model.predict(img_batch, verbose=0)[0]
        keras_top = np.argmax(keras_pred)
        keras_conf = keras_pred[keras_top] * 100
        
        # TFLite prediction
        tflite_interpreter.set_tensor(input_details[0]['index'], img_batch)
        tflite_interpreter.invoke()
        tflite_pred = tflite_interpreter.get_tensor(output_details[0]['index'])[0]
        tflite_top = np.argmax(tflite_pred)
        tflite_conf = tflite_pred[tflite_top] * 100
        
        print(f"\nResults for {os.path.basename(image_path)}:")
        print("-" * 40)
        print(f"Keras prediction:   {class_names[keras_top]} ({keras_conf:.1f}%)")
        print(f"TFLite prediction:  {class_names[tflite_top]} ({tflite_conf:.1f}%)")
        print(f"Predictions match:  {'✅ Yes' if keras_top == tflite_top else '❌ No'}")
        
        # Show top 5 predictions
        print(f"\nTop 5 predictions comparison:")
        keras_top5 = np.argsort(keras_pred)[-5:][::-1]
        tflite_top5 = np.argsort(tflite_pred)[-5:][::-1]
        
        for i in range(5):
            k_idx = keras_top5[i]
            t_idx = tflite_top5[i]
            k_conf = keras_pred[k_idx] * 100
            t_conf = tflite_pred[t_idx] * 100
            
            match_mark = " ✓" if k_idx == t_idx else ""
            print(f"{i+1:2d}. Keras: {class_names[k_idx]:40s} ({k_conf:5.1f}%) | "
                  f"TFLite: {class_names[t_idx]:40s} ({t_conf:5.1f}%){match_mark}")
        
        return {
            'keras_pred': keras_pred,
            'tflite_pred': tflite_pred,
            'match': keras_top == tflite_top
        }
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

def analyze_prediction_distribution(predictions, class_names):
    """Analyze the distribution of predictions"""
    print("\n" + "="*70)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*70)
    
    pred_array = np.array(predictions)
    
    print(f"Shape of predictions: {pred_array.shape}")
    print(f"Sum of predictions: {pred_array.sum():.6f} (should be ~1.0)")
    print(f"Maximum value: {pred_array.max():.6f} ({pred_array.max()*100:.2f}%)")
    print(f"Minimum value: {pred_array.min():.6f} ({pred_array.min()*100:.2f}%)")
    print(f"Mean value: {pred_array.mean():.6f} ({pred_array.mean()*100:.2f}%)")
    print(f"Standard deviation: {pred_array.std():.6f}")
    
    # Check if predictions are uniform (model might be broken)
    uniform_threshold = 0.1  # If max confidence < 10%
    if pred_array.max() < uniform_threshold:
        print(f"⚠️ WARNING: All predictions have low confidence (<{uniform_threshold*100:.0f}%).")
        print("  This could indicate:")
        print("  1. Preprocessing mismatch")
        print("  2. Model not trained properly")
        print("  3. Input is very different from training data")
    
    # Find classes with highest and lowest average predictions
    if len(pred_array.shape) > 1 and pred_array.shape[0] > 1:
        avg_predictions = pred_array.mean(axis=0)
        top_5_idx = np.argsort(avg_predictions)[-5:][::-1]
        bottom_5_idx = np.argsort(avg_predictions)[:5]
        
        print(f"\nClasses with highest average confidence:")
        for idx in top_5_idx:
            print(f"  {class_names[idx]:40s}: {avg_predictions[idx]*100:5.1f}%")
        
        print(f"\nClasses with lowest average confidence:")
        for idx in bottom_5_idx:
            print(f"  {class_names[idx]:40s}: {avg_predictions[idx]*100:5.1f}%")

def verify_preprocessing_pipeline():
    """Verify the preprocessing pipeline matches training"""
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE VERIFICATION")
    print("="*70)
    
    # Create a simple test image
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Save and reload to simulate real image
    cv2.imwrite('temp_test_image.jpg', test_img)
    
    # Method 1: PIL (likely used in training)
    pil_img = Image.open('temp_test_image.jpg')
    pil_array = np.array(pil_img)
    
    print(f"PIL loaded image: shape={pil_array.shape}, dtype={pil_array.dtype}")
    print(f"PIL value range: [{pil_array.min()}, {pil_array.max()}]")
    
    # Method 2: OpenCV (used in web app)
    cv2_img = cv2.imread('temp_test_image.jpg')
    print(f"\nOpenCV loaded image: shape={cv2_img.shape}, dtype={cv2_img.dtype}")
    print(f"OpenCV value range: [{cv2_img.min()}, {cv2_img.max()}]")
    
    # Check if channels are different
    if np.array_equal(pil_array[:,:,0], cv2_img[:,:,2]):
        print("✅ Channel order: PIL uses RGB, OpenCV uses BGR (expected)")
    else:
        print("❌ Channel order mismatch detected!")
    
    # Clean up
    if os.path.exists('temp_test_image.jpg'):
        os.remove('temp_test_image.jpg')

def generate_verification_report(results, class_names):
    """Generate a verification report"""
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    total_tests = len(results)
    matches = sum(1 for r in results.values() if r['match'])
    match_percentage = (matches / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Total tests performed: {total_tests}")
    print(f"Keras-TFLite matches: {matches}/{total_tests} ({match_percentage:.1f}%)")
    
    if match_percentage >= 95:
        print("✅ Excellent match between Keras and TFLite models")
    elif match_percentage >= 80:
        print("⚠️ Good match, minor differences detected")
    elif match_percentage >= 50:
        print("⚠️ Moderate match, significant differences detected")
    else:
        print("❌ Poor match, models may not be equivalent")
    
    print("\nRecommendations:")
    if match_percentage < 95:
        print("1. Check preprocessing pipeline consistency")
        print("2. Verify TFLite conversion settings")
        print("3. Test with more real images from dataset")
        print("4. Check for quantization issues in TFLite")
    else:
        print("1. Models are well-matched, focus on web app preprocessing")
        print("2. Test with actual user images")
        print("3. Verify class names mapping")
    
    # Save report
    report_file = "verification_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL VERIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Keras-TFLite Match Rate: {match_percentage:.1f}%\n\n")
        
        f.write("Test Results:\n")
        f.write("-"*80 + "\n")
        for test_name, result in results.items():
            f.write(f"{test_name:15s}: ")
            f.write(f"Keras={class_names[result['keras_top']]:30s} ({result['keras_conf']:.1f}%) | ")
            f.write(f"TFLite={class_names[result['tflite_top']]:30s} ({result['tflite_conf']:.1f}%) | ")
            f.write(f"Match={'YES' if result['match'] else 'NO'}\n")
        
        f.write(f"\nTotal classes: {len(class_names)}\n")
        f.write(f"Class names file: {'class_names.pkl' if os.path.exists('class_names.pkl') else 'Not found'}\n")
        
        f.write("\nDiagnostic Information:\n")
        f.write("-"*80 + "\n")
        f.write("Common issues and solutions:\n")
        f.write("1. Preprocessing mismatch - Ensure web app matches training preprocessing\n")
        f.write("2. Color channel order - PIL uses RGB, OpenCV uses BGR\n")
        f.write("3. Normalization range - Should be [0, 1] for this model\n")
        f.write("4. Input shape - Must be (1, 128, 128, 3)\n")
        f.write("5. Data type - Must be float32\n")
    
    print(f"\n📄 Detailed report saved to: {report_file}")

def main():
    import time
    
    print("\n" + "="*80)
    print("PLANT DISEASE MODEL VERIFICATION SCRIPT")
    print("="*80)
    print("This script verifies that your Keras and TFLite models produce")
    print("similar predictions and helps diagnose any issues.")
    print("="*80)
    
    # Step 1: Verify model loading
    keras_model, tflite_interpreter = verify_model_loading()
    
    # Step 2: Verify class names
    class_names = verify_class_names()
    
    # Step 3: Verify preprocessing
    verify_preprocessing_pipeline()
    
    # Step 4: Create test inputs
    test_inputs = create_test_inputs()
    
    # Step 5: Compare predictions
    if keras_model is not None and tflite_interpreter is not None:
        results = compare_predictions(keras_model, tflite_interpreter, test_inputs, class_names)
        
        # Step 6: Test with real image if available
        real_image_path = None
        
        # Try to find a real image from the dataset
        possible_paths = [
            "balanced_dataset/train/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG",
            "balanced_dataset/train/Tomato___healthy/0a4c763e-b8d6-48ec-94d5-0920aa9c387e___RS_HL 1805.JPG",
            "cleaned_dataset/train/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                real_image_path = path
                break
        
        if real_image_path:
            real_results = test_with_real_image(real_image_path, keras_model, tflite_interpreter, class_names)
            
            if real_results:
                # Add to results
                results['real_image'] = {
                    'keras_top': np.argmax(real_results['keras_pred']),
                    'keras_conf': np.max(real_results['keras_pred']) * 100,
                    'tflite_top': np.argmax(real_results['tflite_pred']),
                    'tflite_conf': np.max(real_results['tflite_pred']) * 100,
                    'match': real_results['match'],
                    'confidence_diff': abs(np.max(real_results['keras_pred']) * 100 - np.max(real_results['tflite_pred']) * 100)
                }
        
        # Step 7: Analyze prediction distribution
        if keras_model is not None:
            test_pred = keras_model.predict(test_inputs['gray'], verbose=0)[0]
            analyze_prediction_distribution(test_pred, class_names)
        
        # Step 8: Generate report
        generate_verification_report(results, class_names)
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    
    if keras_model is None or tflite_interpreter is None:
        print("\n⚠️ Some models could not be loaded.")
        print("Please ensure:")
        print("1. models/final_model.h5 exists (Keras model)")
        print("2. models/plant_disease_model.tflite exists (TFLite model)")
        print("3. Run model_conversion.py if TFLite model is missing")
    else:
        print("\n✅ Verification script completed successfully.")
        print("Check the generated 'verification_report.txt' for details.")
    
    print("\nNext steps:")
    print("1. If models don't match (>95%), check preprocessing pipeline")
    print("2. Test the fixed web_app.py with sample images")
    print("3. Verify predictions on multiple real images")

if __name__ == "__main__":
    main()