import os
import shutil
import numpy as np

# --- CONFIGURATION ---
# Update this path to point to your TRAIN folder
# Based on your previous paths, it should be:
DATA_DIR = r"StyleClassificationIndoors/StyleClassificationIndoors/train" 

def balance_dataset(data_dir):
    # Check if path exists
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory not found: {data_dir}")
        print("Please check the DATA_DIR path at the top of the script.")
        return

    # 1. Find the target count (the number of images in the biggest folder)
    max_count = 0
    class_counts = {}
    
    # Get list of class folders
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print("📊 Analysing Dataset...")
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        # Only count files, ignore subdirectories
        files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        count = len(files)
        class_counts[cls] = count
        
        if count > max_count:
            max_count = count
        print(f" - {cls}: {count} images")
        
    print(f"\n🎯 Target Count per Class: {max_count}")
    print("="*30)
    
    # 2. Duplicate images in small folders until they match max_count
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        current_count = class_counts[cls]
        
        if current_count < max_count:
            diff = max_count - current_count
            print(f"🔧 Balancing {cls}: Generating {diff} duplicates...")
            
            files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
            
            # Randomly select files to duplicate (with replacement)
            files_to_copy = np.random.choice(files, size=diff, replace=True)
            
            for i, file_name in enumerate(files_to_copy):
                src = os.path.join(cls_path, file_name)
                # Create a unique name for the copy so it doesn't overwrite
                dst = os.path.join(cls_path, f"copy_{i}_{file_name}")
                shutil.copy(src, dst)
                
    print("\n✅ Dataset Balanced! All classes now have equal size.")

if __name__ == "__main__":
    balance_dataset(DATA_DIR)