from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

import data_preprocessing as dp
from model_utils import load_model_any_format

# --- CONFIGURATION ---
MODEL_PATH = Path(__file__).parent / "ResNet_Custom_best.keras"  # <-- change if your model name differs

# Pixel scale for models WITHOUT a built-in Rescaling layer (e.g. ResNet_Custom).
#   False -> feed raw 0-255 pixels (what train.py does)
#   True  -> feed 0-1 pixels (use this ONLY if that model was trained on images divided by 255)
# Models WITH a Rescaling layer inside (e.g. the CCT transformer) always get raw 0-255.
NORMALIZE_IF_NO_RESCALING = False

# --- PAGE SETUP (must be the first Streamlit call) ---
st.set_page_config(page_title="Interior Design Classifier", page_icon="🏠")


# --- CACHING THE MODEL ---
# st.cache_resource makes the model load only once, not on every interaction.
@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        return None, f"Model file `{MODEL_PATH.name}` was not found next to app.py."
    try:
        model = load_model_any_format(MODEL_PATH, compile=False)
        return model, None
    except Exception as e:  # show the real reason instead of hiding it
        return None, f"The model file exists but could not be loaded: {e}"


st.title("🏠 Interior Design Style Classifier")
st.write("Upload a photo of a room, and the AI will detect its design style.")

model, load_error = load_trained_model()

if model is None:
    st.error(load_error)
    st.stop()

has_rescaling = any(isinstance(layer, tf.keras.layers.Rescaling) for layer in model.layers)
scale_to_unit = (not has_rescaling) and NORMALIZE_IF_NO_RESCALING

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Upload a room photo to see the predicted design style.")
    st.stop()

# 1. Display the uploaded image
image = Image.open(uploaded_file)
st.image(image, caption="Uploaded Room", use_container_width=True)

# 2. Preprocess in memory (no temp file, so it is safe with several visitors at once)
processed_img = dp.preprocess_for_model(image, target_size=dp.IMG_SIZE, scale_to_unit=scale_to_unit)
batch = np.expand_dims(processed_img, axis=0)  # (1, 224, 224, 3)

# 3. Predict
with st.spinner("Analyzing..."):
    probs = model.predict(batch, verbose=0)[0]

if len(probs) != len(dp.CLASS_NAMES):
    st.error(
        f"The model outputs {len(probs)} classes but data_preprocessing.CLASS_NAMES has "
        f"{len(dp.CLASS_NAMES)}. Make sure they match."
    )
    st.stop()

predicted_index = int(np.argmax(probs))
confidence = float(probs[predicted_index]) * 100
predicted_style = dp.CLASS_NAMES[predicted_index]

# 4. Show results
st.success(f"**Style Detected:** {predicted_style.replace('-', ' ').upper()}")
st.metric(label="Confidence", value=f"{confidence:.2f}%")

st.write("---")
st.write("**Top 3 Probabilities:**")
for i in np.argsort(probs)[-3:][::-1]:
    score = float(probs[i])
    st.write(f"**{dp.CLASS_NAMES[i].replace('-', ' ').title()}**: {score * 100:.2f}%")
    st.progress(min(max(score, 0.0), 1.0))
