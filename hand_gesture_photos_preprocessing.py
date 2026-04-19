import cv2
import os
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import random
import mediapipe as mp

register_heif_opener()

# MediaPipe hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# 1. Setup Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(BASE_DIR, "hand_gesture_dataset_v3")
output_path = os.path.join(BASE_DIR, "hand_gesture_dataset_processed_9")

# Crop to Hand using MediaPipe
def crop_to_hand(image, use_mediapipe=True):
    """
    Detect hand landmarks and crop image to hand region with padding.
    If no hand detected or use_mediapipe=False, returns original image.
    
    Args:
        image: BGR image from cv2.imread
        use_mediapipe: whether to apply hand detection and cropping
        
    Returns:
        Cropped hand region (square) or original image if no hand detected
    """
    if not use_mediapipe or image is None:
        return image
    
    try:
        h, w, _ = image.shape
        # MediaPipe needs RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Use first detected hand
            hand_lms = results.multi_hand_landmarks[0]
            
            # Get bounding box from hand landmarks
            x_coords = [int(lm.x * w) for lm in hand_lms.landmark]
            y_coords = [int(lm.y * h) for lm in hand_lms.landmark]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add 20% padding
            pad_w = int((x_max - x_min) * 0.2)
            pad_h = int((y_max - y_min) * 0.2)
            
            # Create square crop
            side = max((x_max - x_min) + pad_w, (y_max - y_min) + pad_h)
            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
            
            x1, y1 = max(0, cx - side//2), max(0, cy - side//2)
            x2, y2 = min(w, x1 + side), min(h, y1 + side)
            
            crop = image[y1:y2, x1:x2]
            
            if crop.size > 0:
                return crop
        
        return image  # Return original if no hand detected
    except Exception as e:
        print(f"Warning: Hand detection failed: {e}")
        return image

# 2. Importing and Labeling
images = []
labels = []

# MediaPipe cropping configuration
USE_MEDIAPIPE_CROP = True  # Set to False to skip hand detection/cropping
SKIP_NO_HAND_DETECTED = False  # Set to True to discard images with no hand detected

print(f"Starting Import... (MediaPipe cropping: {USE_MEDIAPIPE_CROP})")
skipped_count = 0

for root, dirs, files in os.walk(raw_dir):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
            path = os.path.join(root, filename)
            label = os.path.basename(root) 
            
            try:
                # Step 1: Load image in color (for MediaPipe detection if enabled)
                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    print(f"Could not read {filename}")
                    continue
                
                # Step 2: Crop to hand if enabled (do this BEFORE resizing for quality)
                if USE_MEDIAPIPE_CROP:
                    img_cropped = crop_to_hand(img_bgr, use_mediapipe=True)
                    if img_cropped is img_bgr:  # No hand was detected
                        if SKIP_NO_HAND_DETECTED:
                            skipped_count += 1
                            continue
                        # else: use full image
                else:
                    img_cropped = img_bgr
                
                # Step 3: Convert to grayscale
                if len(img_cropped.shape) == 3:
                    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = img_cropped
                
                # Step 4: Resize to 50x50
                img_resized = cv2.resize(img_gray, (50, 50))
                
                # Step 5: Histogram equalization
                img_equalized = cv2.equalizeHist(img_resized)
                
                images.append(img_equalized)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if SKIP_NO_HAND_DETECTED:
    print(f"Skipped {skipped_count} images with no hand detected.")

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

print("\n" + "="*60)
print("CONFIGURATION NOTES:")
print("="*60)
print(f"  - USE_MEDIAPIPE_CROP: {USE_MEDIAPIPE_CROP}")
print(f"    Set to True to detect hands and crop before resizing.")
print(f"    This improves image quality vs. cropping already-small images.")
print(f"  - SKIP_NO_HAND_DETECTED: {SKIP_NO_HAND_DETECTED}")
print(f"    Set to True to discard images where no hand was detected.")
print("="*60)


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

                pil = load_img(img_path, color_mode='grayscale', target_size=(50, 50))
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
augment_inplace(output_path, subset='train', augment_probability=0.5)
#
# This file no longer performs augmentation automatically; call the function
# manually from the interactive window so you can control which lines execute.