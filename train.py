from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


# Import your modules
import data_preprocessing as dp
import models_implementation as mi

# --- CONFIGURATION ---
DATA_DIR = str(Path(__file__).parent / "StyleClassificationIndoors" / "StyleClassificationIndoors" / "train")
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 100  # <--- The Long Run

# 1. SETUP GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {gpus}")
    except RuntimeError as e:
        print(e)

# 2. PREPARE DATA
print("⏳ Loading Data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# SAFETY CHECK: the label order the dataset uses must match CLASS_NAMES, otherwise
# the deployed app would show the wrong style names. (Must run before .map/.prefetch.)
if train_ds.class_names != dp.CLASS_NAMES:
    raise ValueError(
        "Class folder names/order do not match data_preprocessing.CLASS_NAMES.\n"
        f"Dataset: {train_ds.class_names}\nCLASS_NAMES: {dp.CLASS_NAMES}"
    )

# Calculate Class Weights (Crucial for imbalance)
class_weights = dp.get_class_weights(DATA_DIR)

# ============================================================
# AGGRESSIVE DATA AUGMENTATION (The Anti-Overfitting Shield)
# ============================================================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    # NOTE: images are on a 0-255 scale here, so a stddev of 0.1 is almost invisible.
    # Raise it (e.g. 5-10) if you really want visible grain.
    tf.keras.layers.GaussianNoise(0.1)
])

print("✨ Aggressive Data Augmentation Configured")

# Apply augmentation to training data only, THEN prefetch (prefetch must be last)
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# 3. BUILD MODEL
model = mi.build_vision_transformer(num_classes=len(dp.CLASS_NAMES))
print(f"🚀 Training Model: {model.name}")

# Compile
# LEARNING RATE 0.0001: Mandatory for CCT/Transformers to prevent crash at start.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. CALLBACKS
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(f"{model.name}_best.keras", save_best_only=True, monitor='val_accuracy'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

# 5. TRAIN
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)


# 6. PLOT RESULTS
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Acc')
    plt.plot(val_acc, label='Validation Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(f"{model.name}_training_plot.png")
    plt.close()  # no plt.show(): it blocks the script until the window is closed


plot_history(history)
print("✅ Training Complete. Model Saved.")
