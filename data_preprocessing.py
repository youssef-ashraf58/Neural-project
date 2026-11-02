import os
import numpy as np

# 1. DEFINE CLASS MAPPING
# NOTE: this order MUST match the alphabetical folder order that
# tf.keras.utils.image_dataset_from_directory uses (train.py checks this).
CLASS_NAMES = [
    'asian', 'boho', 'coastal', 'contemporary', 'craftsman',
    'eclectic', 'farmhouse', 'french-country', 'industrial',
    'mediterranean', 'minimalist', 'modern', 'scandinavian',
    'shabby-chic-style', 'southwestern', 'tropical', 'victorian'
]

# Map string name to integer (e.g., 'asian': 0)
CLASS_INDICES = {name: index for index, name in enumerate(CLASS_NAMES)}

IMG_SIZE = (224, 224)
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')


def get_class_weights(train_dir):
    """
    Scans the training directory to count images per class and calculates
    weights to handle the 'unbalanced data problem'.
    """
    # Imported here so the Streamlit app does not need scikit-learn installed.
    from sklearn.utils.class_weight import compute_class_weight

    y_train = []

    print("   -> Calculating class weights (scanning folders)...")
    for category in CLASS_NAMES:
        path = os.path.join(train_dir, category)
        if os.path.isdir(path):
            num_images = len([f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTENSIONS)])
            y_train.extend([CLASS_INDICES[category]] * num_images)
        else:
            print(f"Warning: Folder for {category} not found in {train_dir}")

    if not y_train:
        raise FileNotFoundError(f"No training images found under: {train_dir}")

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    # Keras expects {class_index: weight}. Using the real class ids (not enumerate)
    # keeps the mapping correct even if a class folder is missing.
    return {int(c): float(w) for c, w in zip(classes, weights)}


def load_and_preprocess_image(image_path, target_size=IMG_SIZE):
    """
    Loads an image from disk, resizes it, and normalizes pixel values to [0, 1].
    (Kept unchanged for the other scripts in the project that use it.)
    """
    import cv2  # imported here so the Streamlit app does not need OpenCV

    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return img


def preprocess_for_model(pil_image, target_size=IMG_SIZE, scale_to_unit=False):
    """
    Preprocesses an in-memory PIL image for prediction, with no temp files.

    Uses tf.image.resize (bilinear), the same resize that
    image_dataset_from_directory applies during training, so the model sees
    images prepared the same way it was trained.

    scale_to_unit=False -> float32 in [0, 255]  (matches train.py)
    scale_to_unit=True  -> float32 in [0, 1]
    """
    import tensorflow as tf
    from PIL import ImageOps

    img = ImageOps.exif_transpose(pil_image).convert("RGB")  # fix phone rotation, drop alpha
    arr = np.asarray(img, dtype=np.float32)
    arr = tf.image.resize(arr, target_size).numpy()
    if scale_to_unit:
        arr = arr / 255.0
    return arr.astype(np.float32)
