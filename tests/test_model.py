#!/usr/bin/env python3
"""Simple verification script:
- Loads saved model `models/model_8/hand_gesture_model_8.keras`
- Prints model input/output shapes
- Runs predictions on up to one sample per class from `hand_gesture_dataset_processed/test`
- Prints predicted label, probability and raw output vector

Run interactively or from terminal.
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent

def load_class_names(base_dir: Path):
    candidates = [
        base_dir / "class_indices_v2.json",
        base_dir / "class_indices_v1.json",
        base_dir / "class_indices.json",
    ]
    for cf in candidates:
        if cf.exists():
            with open(cf, 'r') as f:
                data = json.load(f)
            # if stored as list of names, return directly
            if isinstance(data, list):
                return data
            # if stored as dict mapping, try to convert
            if isinstance(data, dict):
                # expect mapping index->name or name->index
                try:
                    # if values are names
                    return [data[str(i)] for i in range(len(data))]
                except Exception:
                    # fallback to sorting keys
                    return sorted(list(data.keys()))
    # fallback: infer from train folder
    train_dir = base_dir / 'hand_gesture_dataset_processed' / 'train'
    if train_dir.exists():
        names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        if names:
            return names
    return None


def prepare_image(path, size=(220,220)):
    img = Image.open(path).convert('L').resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    return np.expand_dims(arr, 0)  # shape: (1, H, W, 1)


def main(model_path=None, max_samples=8):
    if model_path is None:
        model_path = BASE_DIR / 'models' / 'model_8' / 'hand_gesture_model_8.keras'
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found at {model_path}")

    print('Loading model from:', model_path)
    model = tf.keras.models.load_model(str(model_path))
    print('Model input shape:', model.input_shape)
    print('Model output shape:', model.output_shape)

    class_names = load_class_names(BASE_DIR)
    if class_names is None:
        print('Warning: class names not found; using numeric indices')
        num_out = model.output_shape[-1]
        class_names = [str(i) for i in range(num_out)]
    print('Class names:', class_names)

    test_dir = BASE_DIR / 'hand_gesture_dataset_processed' / 'test'
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # collect up to one example per class
    samples = []
    for name in class_names:
        p = test_dir / name
        if not p.exists() or not p.is_dir():
            continue
        imgs = [f for f in p.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        if imgs:
            samples.append((name, imgs[0]))
        if len(samples) >= max_samples:
            break

    if not samples:
        print('No sample images found in test set')
        return

    print(f'Found {len(samples)} sample(s). Running predictions...')
    for true_label, path in samples:
        batch = prepare_image(path)
        preds = model.predict(batch)
        pred_idx = int(np.argmax(preds, axis=1)[0])
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        prob = float(np.max(preds))
        print(f"File: {path.name}  true={true_label}  predicted={pred_name}  prob={prob:.4f}")
        print('Output vector:', np.round(preds[0], 4).tolist())

    print('\nNote: model.input_shape shows (None, 220, 220, 1).')
    print('`None` is the batch dimension meaning the model accepts any batch size.')
    print('For a single image you supply shape (1,220,220,1) as this script does.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load SavedModel and run quick predictions.')
    parser.add_argument('--model', '-m', help='Path to saved Keras model (.keras folder)', default=None)
    parser.add_argument('--samples', '-n', type=int, default=8, help='Max number of sample images')
    args = parser.parse_args()
    main(model_path=args.model, max_samples=args.samples)
