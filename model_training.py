"""
Plant Disease Recognition - Model Training
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import datetime

def create_lightweight_cnn(input_shape=(128, 128, 3), num_classes=38):
    """
    Create a lightweight CNN model optimized for plant disease recognition
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_simpler_cnn(input_shape=(128, 128, 3), num_classes=38):
    """
    Create an even simpler CNN for faster training
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate settings
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

def train_model(model, train_gen, valid_gen, class_weights=None, epochs=30):
    """
    Train the model with callbacks
    """
    print("\n" + "="*60)
    print("Training Model")
    print("="*60)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        keras.callbacks.ModelCheckpoint(
            filepath='models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        keras.callbacks.TensorBoard(
            log_dir='logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=1
        )
    ]
    
    steps_per_epoch = len(train_gen)
    validation_steps = len(valid_gen)
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Training samples: {train_gen.samples:,}")
    print(f"Validation samples: {valid_gen.samples:,}")
    
    if class_weights:
        print(f"Using class weights")
    
    print("\nStarting training...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_gen,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history

def evaluate_model(model, valid_gen, class_names):
    """
    Evaluate the trained model
    """
    print("\n" + "="*60)
    print("Evaluating Model")
    print("="*60)
    
    evaluation = model.evaluate(valid_gen, verbose=0)
    
    print(f"Test Loss: {evaluation[0]:.4f}")
    print(f"Test Accuracy: {evaluation[1]:.4f}")
    print(f"Test Precision: {evaluation[2]:.4f}")
    print(f"Test Recall: {evaluation[3]:.4f}")
    print(f"Test AUC: {evaluation[4]:.4f}")
    
    print("\nGenerating predictions...")
    y_true = []
    y_pred = []
    
    valid_gen.reset()
    
    for i in range(len(valid_gen)):
        if i % 50 == 0:
            print(f"Processing batch {i+1}/{len(valid_gen)}...")
        
        batch_images, batch_labels = valid_gen[i]
        batch_predictions = model.predict(batch_images, verbose=0)
        
        batch_true = np.argmax(batch_labels, axis=1)
        batch_pred = np.argmax(batch_predictions, axis=1)
        
        y_true.extend(batch_true)
        y_pred.extend(batch_pred)
        
        if len(y_true) >= 2000:
            break
    
    from sklearn.metrics import confusion_matrix, classification_report
    
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    
    display_names = []
    for name in class_names:
        formatted = name.replace('___', ' - ').replace('_', ' ')
        if len(formatted) > 40:
            formatted = formatted[:37] + "..."
        display_names.append(formatted)
    
    report = classification_report(
        y_true[:2000], 
        y_pred[:2000], 
        target_names=display_names,
        digits=3
    )
    print(report)
    
    with open('models/classification_report.txt', 'w') as f:
        f.write(report)
    
    return evaluation, y_true, y_pred

def plot_training_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].plot(history.history['accuracy'], label='Training')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history.history['loss'], label='Training')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(history.history['precision'], label='Training')
    axes[0, 2].plot(history.history['val_precision'], label='Validation')
    axes[0, 2].set_title('Model Precision')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history.history['recall'], label='Training')
    axes[1, 0].plot(history.history['val_recall'], label='Validation')
    axes[1, 0].set_title('Model Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history.history['auc'], label='Training')
    axes[1, 1].plot(history.history['val_auc'], label='Validation')
    axes[1, 1].set_title('Model AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    if 'lr' in history.history:
        axes[1, 2].plot(history.history['lr'], label='Learning Rate')
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Rate')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].axis('off')
    
    plt.suptitle('Training History', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Training history plots saved")

def save_model_summary(model, history, evaluation, filename='models/model_summary.txt'):
    """
    Save model summary to file
    """
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Model Summary\n")
        f.write("="*70 + "\n\n")
        
        f.write("Model Architecture:\n")
        f.write("-"*40 + "\n")
        
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        f.write("\n" + "="*70 + "\n")
        f.write("Training Results\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Final Training Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
        
        f.write("\nTest Set Evaluation:\n")
        f.write("-"*40 + "\n")
        f.write(f"Test Loss: {evaluation[0]:.4f}\n")
        f.write(f"Test Accuracy: {evaluation[1]:.4f}\n")
        f.write(f"Test Precision: {evaluation[2]:.4f}\n")
        f.write(f"Test Recall: {evaluation[3]:.4f}\n")
        f.write(f"Test AUC: {evaluation[4]:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Training Parameters\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Epochs trained: {len(history.history['accuracy'])}\n")
        f.write(f"Training samples: {train_gen.samples:,}\n")
        f.write(f"Validation samples: {valid_gen.samples:,}\n")
        f.write(f"Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Number of classes: {NUM_CLASSES}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Performance Metrics\n")
        f.write("="*70 + "\n\n")
        
        train_acc_min = min(history.history['accuracy'])
        train_acc_max = max(history.history['accuracy'])
        val_acc_min = min(history.history['val_accuracy'])
        val_acc_max = max(history.history['val_accuracy'])
        
        f.write(f"Training Accuracy Range: {train_acc_min:.4f} - {train_acc_max:.4f}\n")
        f.write(f"Validation Accuracy Range: {val_acc_min:.4f} - {val_acc_max:.4f}\n")
        f.write(f"Overfitting Gap: {train_acc_max - val_acc_max:.4f}\n")
    
    print(f"Model summary saved")

def main():
    global train_gen, valid_gen, IMG_SIZE, BATCH_SIZE, NUM_CLASSES
    
    print("="*70)
    print("Plant Disease Recognition - Model Training")
    print("="*70)
    
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    NUM_CLASSES = 38
    EPOCHS = 30
    
    try:
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            horizontal_flip=True,
            zoom_range=0.2
        )
        
        valid_datagen = ImageDataGenerator(rescale=1./255)
        
        train_gen = train_datagen.flow_from_directory(
            'balanced_dataset/train',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        valid_gen = valid_datagen.flow_from_directory(
            'balanced_dataset/valid',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"Loaded {len(class_names)} classes")
        
    except FileNotFoundError:
        print("Preprocessing files not found")
        return
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("\n" + "="*60)
    print("Select Model Architecture")
    print("="*60)
    print("1. Standard CNN")
    print("2. Lightweight CNN (Recommended for CPU)")
    
    choice = input("\nSelect model (1 or 2): ").strip()
    
    if choice == "1":
        model = create_lightweight_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), 
                                      num_classes=NUM_CLASSES)
        print("Selected: Standard CNN")
    else:
        model = create_simpler_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), 
                                  num_classes=NUM_CLASSES)
        print("Selected: Lightweight CNN")
    
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)
    model.summary()
    
    print("\n" + "="*60)
    print("Compiling Model")
    print("="*60)
    model = compile_model(model, learning_rate=0.001)
    print("Model compiled")
    
    history = train_model(model, train_gen, valid_gen, epochs=EPOCHS)
    
    evaluation, y_true, y_pred = evaluate_model(model, valid_gen, class_names)
    
    plot_training_history(history)
    
    model.save('models/final_model.h5')
    print(f"\nModel saved as 'models/final_model.h5'")
    
    save_model_summary(model, history, evaluation)
    
    print("\n" + "="*70)
    print("Training Complete")
    print("="*70)
    
    print(f"\nFinal Results:")
    print(f"  Validation Accuracy: {evaluation[1]:.4f}")
    print(f"  Validation Loss: {evaluation[0]:.4f}")
    print(f"  Precision: {evaluation[2]:.4f}")
    print(f"  Recall: {evaluation[3]:.4f}")
    
    print(f"\nFiles Created:")
    print(f"  models/final_model.h5 - Final trained model")
    print(f"  models/best_model.h5 - Best model")
    print(f"  models/training_history.png - Training graphs")
    print(f"  models/model_summary.txt - Results")
    print(f"  models/classification_report.txt - Performance report")
    
    print(f"\nNext Step:")
    print(f"  Run: python 04_model_conversion.py")

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU warning: {e}")
    
    main()