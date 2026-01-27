"""Data Preprocessing Script for Plant Disease Recognition
"""
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import json
import pickle

def create_tf_datasets(dataset_path, img_size=(128, 128), batch_size=32):
    """
    Create TensorFlow datasets from train/valid folders
    """
    print("="*60)
    print("Creating TensorFlow Datasets")
    print("="*60)
    
    train_dir = os.path.join(dataset_path, 'train')
    valid_dir = os.path.join(dataset_path, 'valid')
    
    print(f"Training data: {train_dir}")
    print(f"Validation data: {valid_dir}")
    print(f"Image size: {img_size[0]}x{img_size[1]}")
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    
    print("\nCreating training generator...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    print("Creating validation generator...")
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(train_generator.class_indices.keys())
    class_indices = train_generator.class_indices
    
    print(f"\nDataset loaded")
    print(f"Classes: {len(class_names)}")
    print(f"Training samples: {train_generator.samples:,}")
    print(f"Validation samples: {valid_generator.samples:,}")
    print(f"Training steps: {len(train_generator)}")
    print(f"Validation steps: {len(valid_generator)}")
    print(f"Batch size: {batch_size}")
    
    return train_generator, valid_generator, class_names, class_indices

def analyze_class_distribution(generator, class_names, split_name):
    """
    Analyze class distribution
    """
    print(f"\n{split_name.upper()} Class Distribution")
    print("-"*40)
    
    class_counts = np.bincount(generator.classes)
    total_samples = sum(class_counts)
    
    print(f"Total samples: {total_samples:,}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Average per class: {total_samples/len(class_names):.0f}")
    
    max_count = max(class_counts)
    min_count = min(class_counts)
    imbalance_ratio = max_count / min_count
    
    print(f"\nBalance Analysis:")
    print(f"Most frequent class: {max_count:,} samples")
    print(f"Least frequent class: {min_count:,} samples")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 2:
        print(f"Class imbalance detected")
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(generator.classes),
            y=generator.classes
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    else:
        print(f"Classes are balanced")
        class_weight_dict = {i: 1.0 for i in range(len(class_names))}
    
    return class_counts, class_weight_dict

def visualize_augmentation(generator, class_names):
    """
    Visualize augmented images
    """
    print("\nVisualizing Data Augmentation")
    print("-"*30)
    
    augmented_images, labels = next(generator)
    label_indices = np.argmax(labels, axis=1)
    
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(min(8, len(augmented_images))):
        ax = axes[i]
        ax.imshow(augmented_images[i])
        
        class_idx = label_indices[i]
        class_name = class_names[class_idx]
        display_name = class_name.replace('___', ' - ').replace('_', ' ')[:30]
        
        ax.set_title(display_name, fontsize=11)
        ax.axis('off')
    
    plt.suptitle("Augmented Training Images", fontsize=14)
    plt.tight_layout()
    plt.savefig('data_augmentation_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Augmentation samples saved")

def save_dataset_info(class_names, class_indices, img_size, batch_size):
    """
    Save dataset information for later use
    """
    os.makedirs("models", exist_ok=True)
    
    plants = {}
    for class_name in class_names:
        if '___' in class_name:
            plant = class_name.split('___')[0]
            disease = class_name.split('___')[1]
            
            if plant not in plants:
                plants[plant] = {
                    'diseases': [],
                    'healthy': False
                }
            
            if 'healthy' in disease.lower():
                plants[plant]['healthy'] = True
                plants[plant]['healthy_class_idx'] = class_indices[class_name]
            else:
                plants[plant]['diseases'].append({
                    'name': disease,
                    'class_idx': class_indices[class_name]
                })
    
    dataset_info = {
        'class_names': class_names,
        'class_indices': class_indices,
        'num_classes': len(class_names),
        'img_height': img_size[0],
        'img_width': img_size[1],
        'batch_size': batch_size,
        'plants': plants,
        'total_plants': len(plants)
    }
    
    info_file = os.path.join("models", 'dataset_info.json')
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    classes_file = os.path.join("models", 'class_names.txt')
    with open(classes_file, 'w') as f:
        f.write("CLASS INDEX MAPPING:\n")
        f.write("="*50 + "\n")
        for class_name, idx in class_indices.items():
            display_name = class_name.replace('___', ' - ').replace('_', ' ')
            f.write(f"{idx:2d}: {display_name}\n")
    
    print(f"\nDataset info saved")
    
    return dataset_info

def plot_class_distribution(class_counts, class_names, split_name):
    """
    Plot class distribution
    """
    plt.figure(figsize=(12, 8))
    
    sorted_indices = np.argsort(class_counts)
    sorted_counts = class_counts[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    
    display_names = []
    for name in sorted_names:
        display_name = name.replace('___', ' - ').replace('_', ' ')
        if len(display_name) > 40:
            display_name = display_name[:37] + "..."
        display_names.append(display_name)
    
    y_pos = np.arange(len(class_counts))
    
    plt.barh(y_pos, sorted_counts, alpha=0.7, color='steelblue')
    plt.yticks(y_pos, display_names, fontsize=9)
    plt.xlabel('Number of Images')
    plt.title(f'Class Distribution - {split_name} Set')
    
    plt.tight_layout()
    
    filename = f'class_distribution_{split_name.lower()}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Class distribution plot saved")

def main():
    DATASET_PATH = "balanced_dataset"
    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    
    print("="*60)
    print("Data Preprocessing")
    print("="*60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*60)
    
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found: {DATASET_PATH}")
        return
    
    train_gen, valid_gen, class_names, class_indices = create_tf_datasets(
        DATASET_PATH, 
        img_size=IMG_SIZE, 
        batch_size=BATCH_SIZE
    )
    
    print("\n" + "="*60)
    print("Class Distribution Analysis")
    print("="*60)
    
    train_counts, train_weights = analyze_class_distribution(train_gen, class_names, "training")
    valid_counts, _ = analyze_class_distribution(valid_gen, class_names, "validation")
    
    plot_class_distribution(train_counts, class_names, "Training")
    plot_class_distribution(valid_counts, class_names, "Validation")
    
    visualize_augmentation(train_gen, class_names)
    
    print("\n" + "="*60)
    print("Saving Dataset Metadata")
    print("="*60)
    
    dataset_info = save_dataset_info(class_names, class_indices, IMG_SIZE, BATCH_SIZE)
    
    with open('class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    
    with open('class_indices.pkl', 'wb') as f:
        pickle.dump(class_indices, f)
    
    print("\n" + "="*60)
    print("Preprocessing Complete")
    print("="*60)
    
    print(f"\nDataset Summary:")
    print(f"Total plants: {dataset_info['total_plants']}")
    print(f"Total classes: {dataset_info['num_classes']}")
    print(f"Training images: {train_gen.samples:,}")
    print(f"Validation images: {valid_gen.samples:,}")
    print(f"Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    
    print(f"\nPlants Covered:")
    plants = dataset_info['plants']
    for i, (plant_name, info) in enumerate(sorted(plants.items())):
        disease_count = len(info['diseases'])
        has_healthy = "Yes" if info['healthy'] else "No"
        print(f"{i+1:2d}. {plant_name:15s}: {disease_count:2d} diseases, Healthy: {has_healthy}")
    
    return train_gen, valid_gen, class_names, train_weights

if __name__ == "__main__":
    train_gen, valid_gen, class_names, class_weights = main()