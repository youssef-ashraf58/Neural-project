import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import gc # Garbage collection

# Import your modules
import data_preprocessing as dp
import models_implementation as mi

# --- CONFIGURATION ---
DATA_DIR = r"StyleClassificationIndoors/StyleClassificationIndoors/train" 
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30 

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

# SAFETY CHECK: label order must match CLASS_NAMES, or the app/test scripts show wrong styles.
if train_ds.class_names != dp.CLASS_NAMES:
    raise ValueError(
        "Class folder names/order do not match data_preprocessing.CLASS_NAMES.\n"
        f"Dataset: {train_ds.class_names}\nCLASS_NAMES: {dp.CLASS_NAMES}"
    )

# 3. APPLY DATA AUGMENTATION
print("✨ Configuring Data Augmentation...")
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# 4. DEFINE MODELS
models_to_train = [
    (mi.build_simple_cnn, "Simple_CNN"),
    (mi.build_deep_cnn, "Deep_CNN"),
    (mi.build_resnet_custom, "ResNet_Custom"),
    (mi.build_vision_transformer, "Vision_Transformer") 
]

# 5. TRAINING LOOP
results = []

for build_func, name in models_to_train:
    # --- SAFETY STEP: CLEAR GPU MEMORY ---
    tf.keras.backend.clear_session()
    gc.collect()
    print("\n" + "="*50)
    print(f"🚀 STARTING TRAINING: {name}")
    print("="*50)

    # Build model
    model = build_func(num_classes=17)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(f"{name}_best.keras", save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    best_acc = max(history.history['val_accuracy'])
    results.append({'Model': name, 'Best_Validation_Acc': best_acc})
    
    # Save Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f'{name} - Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'{name} - Loss')
    plt.legend()
    plt.savefig(f"{name}_training_plot.png")
    plt.close()
    
    print(f"✅ Finished {name}. Best Val Acc: {best_acc:.2%}")
    
    # Delete model from Python memory to be safe
    del model

# 6. SAVE FINAL REPORT
print("\n" + "="*50)
print("🏆 FINAL RESULTS")
print("="*50)
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("training_comparison_results.csv", index=False)