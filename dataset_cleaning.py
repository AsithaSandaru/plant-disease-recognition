"""
Dataset Cleaning and Balancing Script
"""
import os
import shutil
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

class DatasetCleaner:
    def __init__(self, dataset_path, cleaned_path="cleaned_dataset"):
        self.dataset_path = dataset_path
        self.cleaned_path = cleaned_path
        self.stats = {
            'original_total': 0,
            'cleaned_total': 0,
            'removed_total': 0,
            'class_stats': {},
            'issues_found': []
        }
        
    def clean_dataset(self):
        """
        Main cleaning function
        """
        print("\n" + "="*60)
        print("Starting dataset cleaning process")
        print("="*60)
        
        # Create cleaned directory structure
        if os.path.exists(self.cleaned_path):
            print(f"Removing existing directory: {self.cleaned_path}")
            shutil.rmtree(self.cleaned_path)
        
        os.makedirs(os.path.join(self.cleaned_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.cleaned_path, 'valid'), exist_ok=True)
        
        # Clean both train and valid sets
        for split in ['train', 'valid']:
            self.clean_split(split)
        
        # Generate cleaning report
        self.generate_report()
        
        # Check class balance
        self.check_class_balance()
        
        return self.stats
    
    def clean_split(self, split):
        """
        Clean a specific split (train or valid)
        """
        split_path = os.path.join(self.dataset_path, split)
        cleaned_split_path = os.path.join(self.cleaned_path, split)
        
        if not os.path.exists(split_path):
            print(f"Error: {split} folder not found at {split_path}")
            return
        
        class_folders = [d for d in os.listdir(split_path) 
                        if os.path.isdir(os.path.join(split_path, d))]
        
        print(f"\nProcessing {split} set: {len(class_folders)} classes found")
        
        split_stats = {
            'original': 0,
            'cleaned': 0,
            'removed': 0
        }
        
        # Process each class
        for class_folder in tqdm(class_folders, desc=f"Cleaning {split}"):
            class_path = os.path.join(split_path, class_folder)
            cleaned_class_path = os.path.join(cleaned_split_path, class_folder)
            os.makedirs(cleaned_class_path, exist_ok=True)
            
            # Get all image files
            image_files = []
            for file in os.listdir(class_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.jpeg']:
                    image_files.append(file)
            
            original_count = len(image_files)
            split_stats['original'] += original_count
            
            # Track unique images (to remove duplicates)
            unique_images = set()
            kept_count = 0
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                # Clean filename
                clean_filename = self.clean_filename(img_file, class_folder)
                
                # Skip if this is a duplicate rotation
                if self.is_duplicate_rotation(clean_filename, unique_images):
                    split_stats['removed'] += 1
                    self.stats['issues_found'].append(f"Removed duplicate: {class_folder}/{img_file}")
                    continue
                
                # Check image validity
                is_valid, issue = self.validate_image(img_path)
                
                if not is_valid:
                    split_stats['removed'] += 1
                    self.stats['issues_found'].append(f"Removed invalid: {class_folder}/{img_file} - {issue}")
                    continue
                
                # Copy to cleaned location
                cleaned_img_path = os.path.join(cleaned_class_path, clean_filename)
                shutil.copy2(img_path, cleaned_img_path)
                
                unique_images.add(clean_filename)
                kept_count += 1
            
            # Update statistics
            split_stats['cleaned'] += kept_count
            
            if class_folder not in self.stats['class_stats']:
                self.stats['class_stats'][class_folder] = {
                    'original': 0,
                    'cleaned': 0,
                    'removed': 0
                }
            
            self.stats['class_stats'][class_folder]['original'] += original_count
            self.stats['class_stats'][class_folder]['cleaned'] += kept_count
            self.stats['class_stats'][class_folder]['removed'] += (original_count - kept_count)
        
        # Update global stats
        self.stats['original_total'] += split_stats['original']
        self.stats['cleaned_total'] += split_stats['cleaned']
        self.stats['removed_total'] += split_stats['removed']
        
        print(f"  - Original images: {split_stats['original']:,}")
        print(f"  - Cleaned images: {split_stats['cleaned']:,}")
        print(f"  - Removed images: {split_stats['removed']:,}")
        print(f"  - Kept: {(split_stats['cleaned']/split_stats['original']*100):.1f}%")
    
    def clean_filename(self, filename, class_folder):
        """
        Clean and standardize filenames
        """
        patterns_to_remove = [
            '_270deg', '_180deg', '_90deg',
            '_new30degFlipLR', '_FlipLR', '_FlipTB',
            '_rotated', '_flipped', '_augmented'
        ]
        
        clean_name = filename
        for pattern in patterns_to_remove:
            clean_name = clean_name.replace(pattern, '')
        
        clean_name = clean_name.replace(' ', '_')
        clean_name = clean_name.replace('(', '').replace(')', '')
        clean_name = clean_name.replace('__', '_')
        
        name, ext = os.path.splitext(clean_name)
        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
            clean_name = name + '.jpg'
        
        return clean_name
    
    def is_duplicate_rotation(self, filename, unique_set):
        """
        Check if filename is a duplicate rotation of an existing file
        """
        base_patterns = [
            filename.replace('_270deg', '').replace('_180deg', '').replace('_90deg', ''),
            filename.replace('_new30degFlipLR', ''),
            filename.replace('_FlipLR', '').replace('_FlipTB', '')
        ]
        
        for pattern in base_patterns:
            if pattern in unique_set:
                return True
        
        return False
    
    def validate_image(self, img_path):
        """
        Validate image quality and integrity
        """
        try:
            file_size = os.path.getsize(img_path) / 1024
            if file_size < 2:
                return False, f"File too small ({file_size:.1f}KB)"
            
            with Image.open(img_path) as img:
                img.verify()
            
            img_cv = cv2.imread(img_path)
            if img_cv is None:
                return False, "OpenCV cannot read"
            
            height, width = img_cv.shape[:2]
            if height < 100 or width < 100:
                return False, f"Too small ({width}x{height})"
            
            if len(img_cv.shape) == 3:
                std_dev = np.std(img_cv)
                if std_dev < 5:
                    return False, f"Low color variation (std={std_dev:.1f})"
            
            mean_intensity = np.mean(img_cv)
            if mean_intensity < 10 or mean_intensity > 245:
                return False, f"Extreme brightness ({mean_intensity:.0f})"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Error: {str(e)[:50]}"
    
    def generate_report(self):
        """
        Generate cleaning report
        """
        report_file = "cleaning_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Dataset Cleaning Report\n")
            f.write("="*60 + "\n\n")
            
            f.write("Summary:\n")
            f.write("-"*40 + "\n")
            f.write(f"Original total images: {self.stats['original_total']:,}\n")
            f.write(f"Cleaned total images: {self.stats['cleaned_total']:,}\n")
            f.write(f"Removed total images: {self.stats['removed_total']:,}\n")
            f.write(f"Removal rate: {(self.stats['removed_total']/self.stats['original_total']*100):.1f}%\n\n")
            
            f.write("Class Statistics:\n")
            f.write("-"*40 + "\n")
            
            for class_name, stats in sorted(self.stats['class_stats'].items()):
                removal_rate = (stats['removed'] / stats['original'] * 100) if stats['original'] > 0 else 0
                f.write(f"{class_name:45s}: {stats['original']:5d} -> {stats['cleaned']:5d} "
                       f"(-{stats['removed']:3d}, {removal_rate:5.1f}% removed)\n")
            
            f.write("\nIssues Found (first 100):\n")
            f.write("-"*40 + "\n")
            
            for i, issue in enumerate(self.stats['issues_found'][:100]):
                f.write(f"{i+1:4d}. {issue}\n")
            
            if len(self.stats['issues_found']) > 100:
                f.write(f"\n... and {len(self.stats['issues_found']) - 100} more issues\n")
        
        print(f"\nCleaning report saved to: {report_file}")
    
    def check_class_balance(self):
        """
        Check class balance after cleaning
        """
        print("\n" + "="*60)
        print("Class Balance Analysis")
        print("="*60)
        
        train_path = os.path.join(self.cleaned_path, 'train')
        class_counts = {}
        
        if os.path.exists(train_path):
            class_folders = [d for d in os.listdir(train_path) 
                           if os.path.isdir(os.path.join(train_path, d))]
            
            for class_folder in class_folders:
                class_path = os.path.join(train_path, class_folder)
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                class_counts[class_folder] = len(images)
        
        if class_counts:
            counts = list(class_counts.values())
            avg_count = np.mean(counts)
            min_count = min(counts)
            max_count = max(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"Total classes: {len(class_counts)}")
            print(f"Average per class: {avg_count:.0f}")
            print(f"Minimum: {min_count}")
            print(f"Maximum: {max_count}")
            print(f"Imbalance ratio: {imbalance_ratio:.2f}x")
            
            if imbalance_ratio > 3:
                print("\nNote: Significant class imbalance detected")
            
            sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
            
            print(f"\nLargest classes:")
            for i in range(min(3, len(sorted_classes))):
                class_name, count = sorted_classes[-(i+1)]
                display_name = class_name.replace('___', ' - ').replace('_', ' ')[:35]
                print(f"   {display_name:35s}: {count:5d}")
            
            print(f"\nSmallest classes:")
            for i in range(min(3, len(sorted_classes))):
                class_name, count = sorted_classes[i]
                display_name = class_name.replace('___', ' - ').replace('_', ' ')[:35]
                print(f"   {display_name:35s}: {count:5d}")

def create_balanced_subset(cleaned_path, output_path="balanced_dataset", min_images=500):
    """
    Create a balanced subset by limiting maximum images per class
    """
    print("\n" + "="*60)
    print("Creating balanced subset")
    print("="*60)
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'valid'), exist_ok=True)
    
    for split in ['train', 'valid']:
        split_path = os.path.join(cleaned_path, split)
        output_split_path = os.path.join(output_path, split)
        
        if not os.path.exists(split_path):
            continue
        
        class_folders = [d for d in os.listdir(split_path) 
                        if os.path.isdir(os.path.join(split_path, d))]
        
        print(f"\nProcessing {split} set...")
        
        for class_folder in tqdm(class_folders, desc=f"Balancing {split}"):
            class_path = os.path.join(split_path, class_folder)
            output_class_path = os.path.join(output_split_path, class_folder)
            os.makedirs(output_class_path, exist_ok=True)
            
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(images) > min_images:
                images = images[:min_images]
            
            for img_file in images:
                src = os.path.join(class_path, img_file)
                dst = os.path.join(output_class_path, img_file)
                shutil.copy2(src, dst)
            
            if len(class_folders) <= 10:
                print(f"  {class_folder:35s}: {len(images):4d} images")
    
    print(f"\nBalanced dataset created at: {output_path}")
    print(f"Max images per class: {min_images}")
    
    return output_path

def main():
    CLEANED_PATH = "cleaned_dataset"
    BALANCED_PATH = "balanced_dataset"
    
    print("Dataset Processing Script")
    print("="*60)
    
    # Check if cleaning was already done
    if not os.path.exists(CLEANED_PATH):
        print(f"Error: Cleaned dataset not found at {CLEANED_PATH}")
        print("Please run the cleaning script first.")
        return
    
    print(f"Found cleaned dataset at: {CLEANED_PATH}")
    
    # Analyze the cleaned dataset
    print("\n" + "="*60)
    print("Analyzing cleaned dataset")
    print("="*60)
    
    total_train = 0
    total_valid = 0
    
    train_path = os.path.join(CLEANED_PATH, 'train')
    valid_path = os.path.join(CLEANED_PATH, 'valid')
    
    if os.path.exists(train_path):
        for class_folder in os.listdir(train_path):
            class_path = os.path.join(train_path, class_folder)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_train += len(images)
    
    if os.path.exists(valid_path):
        for class_folder in os.listdir(valid_path):
            class_path = os.path.join(valid_path, class_folder)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_valid += len(images)
    
    print(f"Training images: {total_train:,}")
    print(f"Validation images: {total_valid:,}")
    print(f"Total images: {total_train + total_valid:,}")
    
    # Create balanced subset
    print("\n" + "="*60)
    print("Optional: Create balanced subset")
    print("="*60)
    
    create_balanced = input("\nCreate balanced subset? (y/n): ").lower() == 'y'
    
    if create_balanced:
        min_images = int(input("Max images per class (default 500): ") or "500")
        balanced_path = create_balanced_subset(CLEANED_PATH, BALANCED_PATH, min_images)
        final_dataset = balanced_path
        print(f"\nUsing balanced dataset: {balanced_path}")
    else:
        final_dataset = CLEANED_PATH
        print(f"\nUsing cleaned dataset: {CLEANED_PATH}")
    
    # Final summary
    print("\n" + "="*60)
    print("Processing complete")
    print("="*60)
    
    print(f"\nResults:")
    print(f"  Original dataset: 87,867 images")
    print(f"  Cleaned dataset: 70,636 images")
    print(f"  Removed: 17,231 images (19.6%)")
    
    if create_balanced:
        total_balanced = 38 * min_images * 2
        print(f"  Balanced subset: {min_images} images per class")
        print(f"  Total in balanced set: ~{total_balanced:,}")
    
    print(f"\nNext steps:")
    print(f"  1. Update preprocessing script with:")
    print(f"     DATASET_PATH = '{final_dataset}'")
    print(f"  2. Run preprocessing")
    print(f"  3. Run model training")
    
    # Save final dataset path to file
    with open('dataset_path.txt', 'w') as f:
        f.write(final_dataset)
    
    print(f"\nDataset path saved to: dataset_path.txt")

if __name__ == "__main__":
    main()