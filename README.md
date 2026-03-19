# Hand Gesture Recognition (BEng Telerobotics Assignment 3 Tasks 1-3)

Small project to preprocess hand-gesture photos and train a simple CNN to recognize gestures.

## Files

- `hand_gesture_photos_preprocessing.py` — loads raw images from `hand_gesture_dataset_raw/`, standardizes (grayscale, resize 50×50), splits into train/val/test, and writes processed images to `hand_gesture_dataset_processed/`.
- `hand_recognition_model.py` — builds, trains, evaluates, and saves a Keras model. Produces model files and evaluation artifacts.

## Quick start

1. Create a Python environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Preprocess your raw dataset (only if not already done):

```bash
python hand_gesture_photos_preprocessing.py
```

3. Train the model:

```bash
python hand_recognition_model.py
```

Training outputs (saved to project root):

- `hand_gesture_model.keras` — final saved model
- `hand_gesture_model_best.keras` — best model by validation loss
- `class_indices.json` — mapping from class name -> index
- `classification_report.txt`, `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `loss_curve.png`, `accuracy_curve.png`

## Inference example

```python
from tensorflow.keras.models import load_model
import json
import numpy as np
from tensorflow.keras.preprocessing import image

model = load_model('hand_gesture_model.keras')
mapping = json.load(open('class_indices.json'))
inv_map = {v:k for k,v in mapping.items()}

img = image.load_img('some_image.jpg', color_mode='grayscale', target_size=(50,50))
x = image.img_to_array(img)/255.0
x = np.expand_dims(x, 0)
probs = model.predict(x)
label = inv_map[int(np.argmax(probs))]
print(label)
```

## Notes

- On native Windows, TensorFlow >=2.11 does not use CUDA GPUs. Use WSL2 or `tensorflow-directml` for GPU support.
- The preprocessing script writes uint8 images; the training script rescales them with `ImageDataGenerator(rescale=1./255)` so inputs match the network expectation.

If you want, I can add a `requirements.txt` (current environment can be exported), or create a small test harness for inference. Which would you like next?
