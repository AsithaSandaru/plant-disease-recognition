# save as convert_compatible.py
import tensorflow as tf
import os

print(f"Using TensorFlow version: {tf.__version__}")

# Load your model
model = tf.keras.models.load_model('models/best_model.h5')
print("✅ Model loaded")

# Convert with maximum compatibility settings
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# CRITICAL: Force compatibility with older versions
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Use only built-in ops
]

# OPTIONAL: If the above fails, try this instead:
# converter.target_spec.supported_ops = [
#     tf.lite.OpsSet.TFLITE_BUILTINS,
#     tf.lite.OpsSet.SELECT_TF_OPS
# ]

# Disable quantization to ensure compatibility
converter.optimizations = []

print("Converting model with compatibility settings...")
tflite_model = converter.convert()

# Save with a new name
output_path = 'models/plant_disease_model_compatible.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Model saved to {output_path}")
print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

# Verify the converted model
print("\nVerifying model...")
try:
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    print("✅ Model verified - loads successfully!")
    print(f"Input details: {interpreter.get_input_details()[0]['shape']}")
    print(f"Output details: {interpreter.get_output_details()[0]['shape']}")
except Exception as e:
    print(f"❌ Verification failed: {e}")