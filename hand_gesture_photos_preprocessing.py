import cv2
import os
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)
import random
import mediapipe as mp

# Enable HEIC support for Pillow
register_heif_opener()

# MediaPipe hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)

# 1. Setup Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(BASE_DIR, "newdata", "raw")
# Output directory for processed images
output_path = os.path.join(BASE_DIR, "hand_gesture_dataset_processed_12")


# Crop to Hand using MediaPipe
def crop_to_hand(image, use_mediapipe=True):
    """
    Detect hand landmarks and crop image to hand region with padding.
    """
    if not use_mediapipe or image is None:
        return image

    try:
        h, w, _ = image.shape
        # MediaPipe needs RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
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

            x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
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

# Configuration
USE_MEDIAPIPE_CROP = True
SKIP_NO_HAND_DETECTED = False

print(f"Starting Import... (MediaPipe cropping: {USE_MEDIAPIPE_CROP})")
skipped_count = 0

for root, dirs, files in os.walk(raw_dir):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
            path = os.path.join(root, filename)
            label = os.path.basename(root)

            try:
                # --- Step 1: Load image with HEIC support ---
                if filename.lower().endswith(".heic"):
                    # Pillow handles HEIC, then convert to OpenCV BGR
                    pil_img = Image.open(path).convert("RGB")
                    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = cv2.imread(path)

                if img_bgr is None:
                    print(f"Could not read {filename}")
                    continue

                # --- Step 2: Crop to hand ---
                if USE_MEDIAPIPE_CROP:
                    img_cropped = crop_to_hand(img_bgr, use_mediapipe=True)
                    if img_cropped is img_bgr:  # No hand detected
                        if SKIP_NO_HAND_DETECTED:
                            skipped_count += 1
                            continue
                else:
                    img_cropped = img_bgr

                # --- Step 3: Convert to grayscale ---
                if len(img_cropped.shape) == 3:
                    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = img_cropped

                # --- Step 4: Resize and Equalize ---
                img_resized = cv2.resize(img_gray, (50, 50))
                # img_equalized = cv2.equalizeHist(img_resized)

                images.append(img_resized)
                labels.append(label)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if SKIP_NO_HAND_DETECTED:
    print(f"Skipped {skipped_count} images with no hand detected.")

# 3. Splitting the dataset
X_train, X_rest, y_train, y_rest = train_test_split(
    np.array(images), np.array(labels), test_size=0.30, random_state=42, stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.333, random_state=42, stratify=y_rest
)

# 4. Exporting processed images
print(f"\nExporting processed images to {output_path}...")
sets_to_save = [
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test", X_test, y_test),
]

for set_name, images_set, labels_set in sets_to_save:
    for i, (img, lbl) in enumerate(zip(images_set, labels_set)):
        label_dir = os.path.join(output_path, set_name, lbl)
        os.makedirs(label_dir, exist_ok=True)
        cv2.imwrite(os.path.join(label_dir, f"{i}.jpg"), img.astype(np.uint8))

print("\nSuccess! Final Directory Structure created.")


# 5. In-place Augmentation Function
def augment_inplace(processed_base_path, subset="train", augment_probability=0.5):
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=(0.95, 1.05),
        width_shift_range=0.03,
        height_shift_range=0.03,
        shear_range=3.0,
        horizontal_flip=False,
        brightness_range=(0.9, 1.1),
        fill_mode="nearest",
    )

    subset_path = os.path.join(processed_base_path, subset)
    if not os.path.isdir(subset_path):
        return

    print(f"Starting augmentation on {subset_path}...")
    for label in os.listdir(subset_path):
        label_dir = os.path.join(subset_path, label)
        if not os.path.isdir(label_dir):
            continue

        for fname in os.listdir(label_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(label_dir, fname)
                if random.random() > augment_probability:
                    continue
                try:
                    pil = load_img(
                        img_path, color_mode="grayscale", target_size=(50, 50)
                    )
                    x = img_to_array(pil)
                    x_aug = datagen.random_transform(x)
                    x_aug = np.clip(x_aug, 0, 255).astype(np.uint8)
                    Image.fromarray(x_aug.squeeze()).save(img_path)
                except Exception as e:
                    print(f"Augment error for {img_path}: {e}")
    print("Augmentation complete.")


# Trigger augmentation
augment_inplace(output_path, subset="train", augment_probability=0.5)
