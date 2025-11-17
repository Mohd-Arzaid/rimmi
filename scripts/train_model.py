import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.001

print("=" * 50)
print("Egyptian Monument Recognition - Training Script")
print("=" * 50)

# Set paths
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'models/egyptian_monument_model.h5'

# Check if dataset exists
if not os.path.exists(DATASET_DIR):
    print(f"ERROR: Dataset directory '{DATASET_DIR}' not found!")
    print("Please make sure you have images in dataset/pyramids, dataset/sphinx, etc.")
    exit()

# Count images in each category
categories = ['pyramids', 'sphinx', 'karnak', 'luxor', 'abu_simbel']
print("\nDataset Summary:")
print("-" * 50)
for cat in categories:
    cat_path = os.path.join(DATASET_DIR, cat)
    if os.path.exists(cat_path):
        num_images = len([f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"{cat.capitalize():20} : {num_images} images")
    else:
        print(f"{cat.capitalize():20} : FOLDER NOT FOUND!")

print("-" * 50)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,  # 80% train, 20% validation
    fill_mode='nearest'
)

# Validation data (no augmentation)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("\nLoading training data...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("\nLoading validation data...")
validation_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"\nNumber of monument categories: {num_classes}")
print(f"Class mapping: {train_generator.class_indices}")

# Build the model using MobileNetV2
print("\nBuilding model with MobileNetV2 base...")

# Load pre-trained MobileNetV2 (without top classification layer)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model
base_model.trainable = False

# Create custom classification head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
print("-" * 50)
model.summary()
print("-" * 50)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

# Train the model
print(f"\nStarting training for {EPOCHS} epochs...")
print("This will take 15-30 minutes depending on your dataset size...")
print("-" * 50)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 50)
print("Training completed!")
print("=" * 50)

# Save the model
os.makedirs('models', exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# Save class names
class_names = list(train_generator.class_indices.keys())
np.save('models/class_names.npy', class_names)
print(f"Class names saved to: models/class_names.npy")

# Plot training history
print("\nGenerating accuracy and loss graphs...")

plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('models/training_history.png', dpi=150, bbox_inches='tight')
print("Training graphs saved to: models/training_history.png")

# Evaluate on validation set
print("\nEvaluating model on validation data...")
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

# Generate predictions for confusion matrix
print("\nGenerating confusion matrix...")
validation_generator.reset()
predictions = model.predict(validation_generator, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Confusion matrix saved to: models/confusion_matrix.png")

# Classification report
print("\nClassification Report:")
print("-" * 50)
print(classification_report(true_classes, predicted_classes, target_names=class_names))

print("\n" + "=" * 50)
print("✓ Training complete!")
print("=" * 50)
print("\nNext steps:")
print("1. Check models/training_history.png for accuracy graphs")
print("2. Check models/confusion_matrix.png for prediction analysis")
print("3. Model is saved at:", MODEL_SAVE_PATH)
print("4. Ready to convert to TensorFlow.js for web deployment!")
print("=" * 50)