import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# PATHS (RELATIVE - FIXED)
train_dir = "Pomegranate Diseases Dataset/Pomegranate Dataset/training"
val_dir = "Pomegranate Diseases Dataset/Pomegranate Dataset/validation"
test_dir = "Pomegranate Diseases Dataset/Pomegranate Dataset/testing"

IMG_SIZE = 224
BATCH_SIZE = 32

# AUGMENTED GENERATORS (IMPROVED FOR IMBALANCE/ROBUSTNESS)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# DATA LOADERS
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # For eval
)

# CLASS WEIGHTS FOR IMBALANCE (CRITICAL FIX)
classes = np.unique(train_data.classes)
class_weights = compute_class_weight('balanced', classes=classes, y=train_data.classes)
class_weight_dict = {int(c): w for c, w in zip(classes, class_weights)}
print("✅ Class weights computed:", class_weight_dict)
print(f"✅ Dataset loaded: {train_data.samples} train, {val_data.samples} val, {test_data.samples} test images")

# MODEL (EfficientNetB0)
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Stage 1: Frozen

# HEAD
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(train_data.num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# STAGE 1 COMPILE & TRAIN
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🚀 Stage 1: Training frozen base...")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=12,
    class_weight=class_weight_dict
)

# STAGE 2: FINE-TUNE TOP LAYERS
base_model.trainable = True
for layer in base_model.layers[:-25]:  # Unfreeze last 25 layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Low LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🔥 Stage 2: Fine-tuning...")
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=12,
    class_weight=class_weight_dict
)

# FINAL TEST EVAL (CRITICAL)
print("📊 Final Test Evaluation:")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

# SAVE IMPROVED MODEL
model.save("improved_efficientnet.h5")
print("🎉 Improved model saved as 'improved_efficientnet.h5' - use in app.py!")
print("Run: streamlit run \"Pomegranate Diseases Dataset/app.py\" to test!")
