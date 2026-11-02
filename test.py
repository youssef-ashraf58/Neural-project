import tensorflow as tf
import numpy as np
import pandas as pd
import os

# Import preprocessing for image sizing and class names
import data_preprocessing as dp
from model_utils import load_model_any_format

# --- CONFIGURATION ---
# UPDATE THIS to "Deep_CNN_best.keras" if that was your winner in the CSV
MODEL_NAME = "ResNet_Custom_best.keras"  
TEST_DATA_DIR = r"StyleClassificationIndoors/StyleClassificationIndoors/test" 
SAMPLE_CSV_FILE = "sample_submission.csv"
OUTPUT_CSV_FILE = "final_submission.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 1. SETUP GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected: {gpus}")
    except RuntimeError as e:
        print(e)

# 2. LOAD DATA FROM CSV
print(f"⏳ Reading {SAMPLE_CSV_FILE}...")
try:
    df = pd.read_csv(SAMPLE_CSV_FILE)
    print(f"   Found {len(df)} images to process.")
except FileNotFoundError:
    print(f"❌ Error: Could not find {SAMPLE_CSV_FILE}.")
    exit()

# 3. PREPARE IMAGE PATHS
image_paths = [os.path.join(TEST_DATA_DIR, fname) for fname in df['ImageName']]

# 4. CREATE ROBUST DATASET PIPELINE (Crash-Proof Version)
def safe_load_image(path):
    # This runs in pure Python/NumPy mode to avoid TensorFlow graph crashes
    path_str = path.numpy().decode("utf-8")
    try:
        # Read file
        img_raw = tf.io.read_file(path_str)
        # Decode (Handles JPG, PNG, BMP)
        img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
        
        # Resize
        img = tf.image.resize(img, IMG_SIZE)
        
        # Cast to float but DO NOT divide by 255 (Matches your Training)
        img = tf.cast(img, tf.float32)
        
        # CRITICAL: Return as NumPy array to keep TensorFlow happy
        return img.numpy()
        
    except Exception as e:
        # If loading fails, print warning and return black image
        print(f"⚠️ Corrupt file replaced: {os.path.basename(path_str)}")
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)

def process_path_wrapper(path):
    # Force CPU execution for the loading step
    with tf.device('/CPU:0'):
        return tf.py_function(safe_load_image, [path], tf.float32)

print("⏳ Creating data pipeline (Safety Mode)...")

# Force dataset creation on CPU
with tf.device('/CPU:0'):
    test_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    test_ds = test_ds.map(process_path_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

# Restore shape info (lost by py_function)
def set_shape(img):
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img

test_ds = test_ds.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# 5. LOAD MODEL
print(f"⏳ Loading model: {MODEL_NAME}...")
try:
    model = load_model_any_format(MODEL_NAME)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# 6. PREDICT
print("🚀 Generating predictions...")
predictions = model.predict(test_ds, verbose=1)

# 7. PROCESS RESULTS
print("📝 Updating CSV...")
predicted_indices = np.argmax(predictions, axis=1)

# Map indices to Class Names
predicted_labels = [dp.CLASS_NAMES[i] for i in predicted_indices]

# Update DataFrame
df['ClassLabel'] = predicted_labels

# 8. SAVE
df.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"✅ Success! Saved predictions to {OUTPUT_CSV_FILE}")
print(df.head())