import cv2
import os
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from sklearn.model_selection import train_test_split

register_heif_opener()

# 1. Setup Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(BASE_DIR, "hand_gesture_dataset_raw")
output_path = os.path.join(BASE_DIR, "hand_gesture_dataset_processed")

# 2. Importing and Labeling
images = []
labels = []

print("Starting Import...")
for root, dirs, files in os.walk(raw_dir):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
            path = os.path.join(root, filename)
            label = os.path.basename(root) 
            
            try:
                # Step 3: Standardizing (Grayscale, Resize, Normalize)
                pil_img = Image.open(path).convert('L') 
                img_array = np.array(pil_img)
                img_resized = cv2.resize(img_array, (50, 50))
                
                # Normalizing pixel values to (0,1) range
                img_normalized = img_resized / 255.0
                
                images.append(img_normalized)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 4. Splitting the dataset (70% Train, 20% Val, 10% Test)
X_train, X_rest, y_train, y_rest = train_test_split(
    np.array(images), np.array(labels), test_size=0.30, random_state=42, stratify=labels 
)

X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.333, random_state=42, stratify=y_rest
)

# # 5. Preprocessing (Data Augmentation) - Applied ONLY to Train Set but at the moment I have not implemented any augmentation techniques, so I will just duplicate the training data to increase its size by 40%
# X_train_final = np.concatenate((X_train, X_train))
# y_train_final = np.concatenate((y_train, y_train))

# --- VISUAL EXPORT: Saving to Respective Directories ---
print(f"\nExporting processed images to {output_path}...")

sets_to_save = [
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test", X_test, y_test)
]

for set_name, images_set, labels_set in sets_to_save:
    for i, (img, lbl) in enumerate(zip(images_set, labels_set)):
        # Create directory for the specific label within the set: e.g., processed/train/rock/
        label_dir = os.path.join(output_path, set_name, lbl)
        os.makedirs(label_dir, exist_ok=True)
        
        # Convert back to 0-255 for saving
        save_img = img.astype(np.uint8)
        filename = f"{i}.jpg"
        cv2.imwrite(os.path.join(label_dir, filename), save_img)

print("\nSuccess! Final Directory Structure:")
print(f"{output_path}/")
print("  ├── train/ (70%)")
print("  ├── val/   (20%)")
print("  └── test/  (10%)")