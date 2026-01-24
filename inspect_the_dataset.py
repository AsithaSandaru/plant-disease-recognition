"""
Dataset Inspection Script
Checks the new Plant Diseases Dataset structure
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def inspect_dataset_structure(dataset_path):
    """
    Inspect the dataset folder structure
    """
    print("="*70)
    print("🔍 DATASET STRUCTURE INSPECTION")
    print("="*70)
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please download from: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        return None
    
    print(f"📁 Dataset location: {os.path.abspath(dataset_path)}")
    
    # Check main folders
    folders = ['train', 'valid']
    stats = {}
    
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        
        if os.path.exists(folder_path):
            print(f"\n📂 {folder.upper()} folder found")
            
            # Count subfolders (classes)
            classes = [d for d in os.listdir(folder_path) 
                      if os.path.isdir(os.path.join(folder_path, d))]
            
            print(f"  Number of classes: {len(classes)}")
            
            # Count images per class
            class_stats = {}
            total_images = 0
            
            print("  Class distribution:")
            for class_name in tqdm(classes, desc=f"Scanning {folder}"):
                class_path = os.path.join(folder_path, class_name)
                
                # Count image files
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jpeg'))]
                
                image_count = len(images)
                class_stats[class_name] = image_count
                total_images += image_count
            
            stats[folder] = {
                'total_images': total_images,
                'class_stats': class_stats,
                'num_classes': len(classes)
            }
            
            print(f"  Total images: {total_images:,}")
            print(f"  Avg images per class: {total_images/len(classes):.0f}")
            
            # Show smallest and largest classes
            sorted_classes = sorted(class_stats.items(), key=lambda x: x[1])
            print(f"\n  Smallest classes:")
            for i in range(min(3, len(sorted_classes))):
                print(f"    {sorted_classes[i][0]}: {sorted_classes[i][1]} images")
            
            print(f"\n  Largest classes:")
            for i in range(min(3, len(sorted_classes))):
                print(f"    {sorted_classes[-(i+1)][0]}: {sorted_classes[-(i+1)][1]} images")
        else:
            print(f"\n❌ {folder} folder not found!")
    
    return stats

def check_image_properties(dataset_path, sample_size=5):
    """
    Check image properties (size, format, etc.)
    """
    print("\n" + "="*70)
    print("🖼️ IMAGE PROPERTIES CHECK")
    print("="*70)
    
    train_path = os.path.join(dataset_path, 'train')
    
    if not os.path.exists(train_path):
        print("Train folder not found!")
        return
    
    # Get first few classes
    classes = os.listdir(train_path)[:3]
    
    image_properties = []
    
    for class_name in classes:
        class_path = os.path.join(train_path, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:sample_size]
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            
            try:
                # Read image
                img = cv2.imread(img_path)
                
                if img is not None:
                    properties = {
                        'class': class_name,
                        'filename': img_name,
                        'shape': img.shape,
                        'size_kb': os.path.getsize(img_path) / 1024,
                        'dtype': img.dtype,
                        'mean_intensity': np.mean(img)
                    }
                    image_properties.append(properties)
                    
                    print(f"  {class_name}/{img_name}: {img.shape}, {properties['size_kb']:.1f}KB")
            except Exception as e:
                print(f"  ❌ Error reading {img_name}: {e}")
    
    return image_properties

def visualize_sample_images(dataset_path, num_classes=8, num_images=4):
    """
    Display sample images from the dataset
    """
    print("\n" + "="*70)
    print("📸 SAMPLE IMAGES VISUALIZATION")
    print("="*70)
    
    train_path = os.path.join(dataset_path, 'train')
    
    if not os.path.exists(train_path):
        return
    
    classes = sorted([d for d in os.listdir(train_path) 
                     if os.path.isdir(os.path.join(train_path, d))])[:num_classes]
    
    fig, axes = plt.subplots(num_classes, num_images, figsize=(15, 3*num_classes))
    
    if num_classes == 1:
        axes = [axes]
    
    for i, class_name in enumerate(classes):
        class_path = os.path.join(train_path, class_name)
        
        # Get sample images
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:num_images]
        
        for j in range(num_images):
            if j < len(images):
                img_path = os.path.join(class_path, images[j])
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    ax = axes[i][j] if num_classes > 1 else axes[j]
                    ax.imshow(img_rgb)
                    
                    # Shorten class name for display
                    display_name = class_name.replace('___', ' - ').replace('_', ' ')
                    if j == 0:
                        ax.set_ylabel(display_name[:20], fontsize=9, rotation=0, ha='right')
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # Add border
                    for spine in ax.spines.values():
                        spine.set_edgecolor('gray')
                        spine.set_linewidth(0.5)
            else:
                axes[i][j].axis('off')
    
    plt.suptitle(f'Sample Images from {num_classes} Plant Disease Classes', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Sample images saved as 'dataset_samples.png'")

def generate_dataset_report(stats, dataset_path):
    """
    Generate a detailed dataset report
    """
    report_file = "dataset_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PLANT DISEASE DATASET ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset Path: {os.path.abspath(dataset_path)}\n")
        f.write(f"Source: Kaggle - New Plant Diseases Dataset\n")
        f.write(f"URL: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset\n\n")
        
        for split in ['train', 'valid']:
            if split in stats:
                f.write(f"{split.upper()} SET:\n")
                f.write(f"-"*40 + "\n")
                f.write(f"Total images: {stats[split]['total_images']:,}\n")
                f.write(f"Number of classes: {stats[split]['num_classes']}\n")
                f.write(f"Average images per class: {stats[split]['total_images']/stats[split]['num_classes']:.0f}\n\n")
                
                f.write("Class Distribution:\n")
                for class_name, count in sorted(stats[split]['class_stats'].items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / stats[split]['total_images']) * 100
                    f.write(f"  {class_name:40s}: {count:5d} ({percentage:5.1f}%)\n")
                f.write("\n")
        
        # Calculate totals
        total_images = stats['train']['total_images'] + stats['valid']['total_images']
        total_classes = max(stats['train']['num_classes'], stats['valid']['num_classes'])
        
        f.write("OVERALL SUMMARY:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total images in dataset: {total_images:,}\n")
        f.write(f"Total classes: {total_classes}\n")
        f.write(f"Train/Validation split: {stats['train']['total_images']/total_images*100:.1f}% / {stats['valid']['total_images']/total_images*100:.1f}%\n")
        
        # List all classes
        f.write("\nALL CLASSES (38 total):\n")
        f.write("-"*40 + "\n")
        
        classes = sorted(stats['train']['class_stats'].keys())
        plants = set()
        
        for class_name in classes:
            # Extract plant name (first word before ___)
            if '___' in class_name:
                plant = class_name.split('___')[0]
                plants.add(plant)
            
            # Format for display
            display_name = class_name.replace('___', ' - ').replace('_', ' ')
            f.write(f"  • {display_name}\n")
        
        f.write(f"\nPlants covered: {len(plants)}\n")
        f.write(f"Plant species: {', '.join(sorted(plants))}\n")
    
    print(f"\n📊 Detailed report saved to: {report_file}")

def main():
    # Path to your downloaded dataset
    dataset_path = "New Plant Diseases Dataset(Augmented)"  # Change this if needed
    
    print("Plant Disease Recognition - Dataset Inspection")
    print("Dataset: New Plant Diseases Dataset (Kaggle)")
    print("="*70)
    
    # 1. Inspect structure
    stats = inspect_dataset_structure(dataset_path)
    
    if stats is None:
        return
    
    # 2. Check image properties
    image_props = check_image_properties(dataset_path)
    
    # 3. Visualize samples
    visualize_sample_images(dataset_path, num_classes=6, num_images=4)
    
    # 4. Generate report
    generate_dataset_report(stats, dataset_path)
    
    # 5. Recommendations
    print("\n" + "="*70)
    print("✅ DATASET READY FOR TRAINING")
    print("="*70)
    print("\nKey findings:")
    print(f"• {stats['train']['num_classes']} disease classes")
    print(f"• {stats['train']['total_images']:,} training images")
    print(f"• {stats['valid']['total_images']:,} validation images")
    print(f"• 14 plant species covered")
    print("\nNext step: Run 02_preprocessing.py")

if __name__ == "__main__":
    main()