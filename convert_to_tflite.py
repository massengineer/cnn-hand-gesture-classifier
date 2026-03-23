import argparse
import os
import numpy as np
from PIL import Image
import tensorflow as tf


def representative_data_generator(representative_dir, img_size=(50, 50), max_samples=100):
    files = []
    for root, _, filenames in os.walk(representative_dir):
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                files.append(os.path.join(root, f))
    files = files[:max_samples]
    for f in files:
        img = Image.open(f).convert('L').resize(img_size)
        arr = np.asarray(img).astype('float32') / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))  # [1, H, W, 1]
        yield [arr]


def convert(model_path, out_path, representative_dir=None):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if representative_dir and os.path.isdir(representative_dir):
        converter.representative_dataset = lambda: representative_data_generator(representative_dir)
        # Request full integer quantization for better edge performance
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    else:
        # Fall back to float16 quantization if no representative data provided
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert Keras model to TFLite')
    parser.add_argument('--model', required=True, help='Path to .keras or SavedModel')
    parser.add_argument('--out', default='model.tflite', help='Output tflite filename')
    parser.add_argument('--rep', default=None, help='Representative images directory (optional)')
    args = parser.parse_args()
    convert(args.model, args.out, args.rep)


if __name__ == '__main__':
    main()
