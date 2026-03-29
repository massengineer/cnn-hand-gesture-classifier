import cv2
import os
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import random

register_heif_opener()

# 1. Setup Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(BASE_DIR, "hand_gesture_dataset_v2")
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
                # Step 3: Standardizing (Grayscale, Resize)
                pil_img = Image.open(path).convert('L') 
                img_array = np.array(pil_img)
                img_resized = cv2.resize(img_array, (220, 220))
                
                images.append(img_resized)
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
        
        # Save as 0-255 for visualization (model will normalize later)
        save_img = img.astype(np.uint8)
        filename = f"{i}.jpg"
        cv2.imwrite(os.path.join(label_dir, filename), save_img)

print("\nSuccess! Final Directory Structure:")
print(f"{output_path}/")
print("  ├── train/ (70%)")
print("  ├── val/   (20%)")
print("  └── test/  (10%)")


def augment_inplace(processed_base_path, subset='train', augment_probability=0.5,
                    rotation_range=10, zoom_range=(0.95, 1.05), width_shift_range=0.03,
                    height_shift_range=0.03, shear_range=3.0, horizontal_flip=False,
                    brightness_range=(0.9, 1.1)):
    """Apply random augmentations in-place to images inside `processed_base_path/subset`.

    - `augment_probability`: fraction of images to modify (0.0 - 1.0)
    - All augmentations are applied using Keras' ImageDataGenerator's random transforms.
    - `horizontal_flip` is False by default to avoid flipped variants.
    This function overwrites the original files.
    """
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        horizontal_flip=horizontal_flip,
        brightness_range=brightness_range,
        fill_mode='nearest'
    )

    subset_path = os.path.join(processed_base_path, subset)
    if not os.path.isdir(subset_path):
        print(f"Subset path not found: {subset_path}")
        return

    print(f"Starting in-place augmentation on {subset_path} (p={augment_probability})...")
    for label in os.listdir(subset_path):
        label_dir = os.path.join(subset_path, label)
        if not os.path.isdir(label_dir):
            continue

        for fname in os.listdir(label_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(label_dir, fname)
            try:
                if random.random() > augment_probability:
                    continue

                pil = load_img(img_path, color_mode='grayscale', target_size=(220, 220))
                x = img_to_array(pil)
                # datagen.random_transform expects shape (h, w, c)
                x_aug = datagen.random_transform(x)
                x_aug = np.clip(x_aug, 0, 255).astype(np.uint8)
                arr = x_aug.squeeze()
                Image.fromarray(arr).save(img_path)
            except Exception as e:
                print(f"In-place augment error for {img_path}: {e}")

    print("In-place augmentation complete.")


# Usage (interactive):
# To run augmentation from an interactive window or notebook, select and execute
# the following line (or adjust parameters) instead of running the whole script:
#
# augment_inplace(output_path, subset='train', augment_probability=0.5)
#
# This file no longer performs augmentation automatically; call the function
# manually from the interactive window so you can control which lines execute.