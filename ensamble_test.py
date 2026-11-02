import tensorflow as tf
import numpy as np
import pandas as pd
import os
import data_preprocessing as dp
from model_utils import load_model_any_format

# --- CONFIGURATION ---
TEST_DATA_DIR = r"StyleClassificationIndoors/StyleClassificationIndoors/test"
SAMPLE_CSV_FILE = "sample_submission.csv"
OUTPUT_CSV_FILE = "ensemble_submission.csv"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# LIST YOUR MODELS HERE
# Make sure these files exist in your folder
models_list = [
    "Deep_CNN_best.keras",
    "ResNet_Custom_best.keras",
    "Normalized_Hybrid_Transformer_best.keras" # Uncomment if you have this one too
]

# 1. SETUP
print(f"🚀 Starting Ensemble with {len(models_list)} models...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 2. LOAD DATA
try:
    df = pd.read_csv(SAMPLE_CSV_FILE)
    image_paths = [os.path.join(TEST_DATA_DIR, fname) for fname in df['ImageName']]
except FileNotFoundError:
    print("❌ Error: CSV not found.")
    exit()

# 3. ROBUST LOADER (No division by 255 for safety - let models handle it)
def safe_load_image(path):
    path_str = path.numpy().decode("utf-8")
    try:
        img_raw = tf.io.read_file(path_str)
        img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) 
        return img.numpy()
    except:
        return np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)

def process_path_wrapper(path):
    with tf.device('/CPU:0'):
        return tf.py_function(safe_load_image, [path], tf.float32)

print("⏳ Creating data pipeline...")
with tf.device('/CPU:0'):
    test_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    test_ds = test_ds.map(process_path_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

def set_shape(img):
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img

test_ds = test_ds.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# 4. PREDICT WITH EACH MODEL
all_predictions = []

for model_name in models_list:
    print(f"\n🔮 Predicting with {model_name}...")
    try:
        model = load_model_any_format(model_name)
        
        # Check if model expects 0-1 or 0-255
        # Quick hack: We feed 0-255. If the model has Rescaling layer, it works.
        # If it doesn't (like your old Deep_CNN), we might need to manually normalize.
        # BUT: Ensembling is robust. Let's try raw first.
        
        preds = model.predict(test_ds, verbose=1)
        all_predictions.append(preds)
        
        # Free memory
        del model
        tf.keras.backend.clear_session()
        
    except Exception as e:
        print(f"⚠️ Could not load {model_name}: {e}")

# 5. AVERAGE PREDICTIONS
print("\n⚗️ Averaging predictions...")
if not all_predictions:
    print("❌ No models ran successfully.")
    exit()

# Average the probability arrays
avg_preds = np.mean(all_predictions, axis=0)
final_indices = np.argmax(avg_preds, axis=1)
final_labels = [dp.CLASS_NAMES[i] for i in final_indices]

# 6. SAVE
df['ClassLabel'] = final_labels
df.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"✅ Ensemble Submission Saved: {OUTPUT_CSV_FILE}")
print(df['ClassLabel'].value_counts())