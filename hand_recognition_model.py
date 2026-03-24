import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "hand_gesture_dataset_processed")

# Load data using flow_from_directory (automatically labels based on folder names)
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "val")
test_dir = os.path.join(dataset_path, "test")

# Load data in batches
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    target_size=(50, 50),
    batch_size=32,
    color_mode='grayscale',
    label_mode='categorical',
    shuffle=True
)

val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    target_size=(50, 50),
    batch_size=32,
    color_mode='grayscale',
    label_mode='categorical',
    shuffle=False
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    target_size=(50, 50),
    batch_size=32,
    color_mode='grayscale',
    label_mode='categorical',
    shuffle=False
)

# Determine the number of classes from the training data
num_classes = len(train_data.class_indices) 

# Save class indices for later inference (index -> label mapping)
with open(os.path.join(BASE_DIR, "class_indices.json"), "w") as f:
    json.dump(train_data.class_indices, f)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(50, 50, 1)),  # Normalize pixel values to [0,1]
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Display the architecture
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks: save best model and early stop on no improvement
checkpoint_path = os.path.join(BASE_DIR, "hand_gesture_model_best.keras")
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
callbacks = [checkpoint, earlystop]

print("\nTraining the model...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test data
print("\nEvaluating on test data...")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict on the entire test set (test_data.shuffle=False to keep order)
y_true = test_data.classes
preds = model.predict(test_data, verbose=1)
y_pred = preds.argmax(axis=1)

# Print and save confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
cr = classification_report(y_true, y_pred, target_names=[k for k, v in sorted(train_data.class_indices.items(), key=lambda x: x[1])])
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", cr)
with open(os.path.join(BASE_DIR, "classification_report.txt"), "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Report:\n")
    f.write(cr)

# Plot confusion matrix (raw counts) and normalized version
class_names = [k for k, v in sorted(train_data.class_indices.items(), key=lambda x: x[1])]
tick_marks = np.arange(len(class_names))

# Raw confusion matrix heatmap
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(tick_marks, class_names, rotation=45, ha='right')
plt.yticks(tick_marks, class_names)
thresh = cm.max() / 2.0 if cm.size else 0
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, f"{cm[i, j]:d}", ha='center',
            color='white' if cm[i, j] > thresh else 'black')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
plt.close()

# Normalized confusion matrix (rows sum to 1)
cm_norm = cm.astype('float')
row_sums = cm_norm.sum(axis=1)[:, np.newaxis]
with np.errstate(divide='ignore', invalid='ignore'):
    cm_norm = np.divide(cm_norm, row_sums)
cm_norm = np.nan_to_num(cm_norm)

plt.figure(figsize=(8, 6))
plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix')
plt.colorbar()
plt.xticks(tick_marks, class_names, rotation=45, ha='right')
plt.yticks(tick_marks, class_names)
thresh = cm_norm.max() / 2.0 if cm_norm.size else 0
for i, j in np.ndindex(cm_norm.shape):
    plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha='center',
            color='white' if cm_norm[i, j] > thresh else 'black')
plt.ylabel('True label')
plt.xlabel('Predicted label (normalized)')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix_normalized.png'))
plt.close()

# Plot training curves
plt.figure()
plt.plot(history.history.get('loss', []), label='train_loss')
plt.plot(history.history.get('val_loss', []), label='val_loss')
plt.legend()
plt.title('Loss')
plt.savefig(os.path.join(BASE_DIR, 'loss_curve.png'))
plt.close()

plt.figure()
plt.plot(history.history.get('accuracy', []), label='train_acc')
plt.plot(history.history.get('val_accuracy', []), label='val_acc')
plt.legend()
plt.title('Accuracy')
plt.savefig(os.path.join(BASE_DIR, 'accuracy_curve.png'))
plt.close()

# Save the trained model
model_save_path = os.path.join(BASE_DIR, "hand_gesture_model.keras")
model.save(model_save_path)
print(f"\nModel saved to {model_save_path}")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)