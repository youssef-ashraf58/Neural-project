import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import tempfile
from io import BytesIO
from pathlib import Path
from PIL import Image

import data_preprocessing as dp
from model_utils import load_model_any_format

# --- CONFIG ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

st.set_page_config(page_title="Interior Style Classifier", page_icon="🏠", layout="wide")
st.title("🏠 Interior Design Style Classifier")
st.markdown("### Practical Exam Tool")

# ----------------------------------------
# 1) MODEL UPLOAD
# ----------------------------------------
st.sidebar.header("Upload Model (.keras)")
uploaded_model_file = st.sidebar.file_uploader("Upload your .keras model", type=["keras", "h5"])

@st.cache_resource
def load_uploaded_model(file_name, file_bytes):
    # Keep the original extension (.keras / .h5) and use a temp folder, so the
    # model is not written into the project folder. load_model_any_format also
    # handles old HDF5 files that were saved with a .keras name.
    with tempfile.TemporaryDirectory() as tmp:
        temp_path = Path(tmp) / (Path(file_name).name or "uploaded_model.keras")
        temp_path.write_bytes(file_bytes)
        return load_model_any_format(temp_path)

model = None
if uploaded_model_file is not None:
    try:
        model = load_uploaded_model(uploaded_model_file.name, uploaded_model_file.getvalue())
    except Exception as e:
        st.sidebar.error(f"Could not load the model: {e}")

# ----------------------------------------
# Prepare image loader
# ----------------------------------------
def load_image_from_bytes(file):
    # Same preprocessing as training: bilinear resize, raw 0-255 float pixels.
    return dp.preprocess_for_model(Image.open(file), target_size=IMG_SIZE, scale_to_unit=False)

def create_dataset_from_uploaded(uploaded_files):
    imgs = []
    fnames = []
    for uf in uploaded_files:
        img = load_image_from_bytes(uf)
        imgs.append(img)
        fnames.append(uf.name)

    imgs = np.stack(imgs)
    return imgs, fnames

# ----------------------------------------
# TABS
# ----------------------------------------
tab1, tab2 = st.tabs(["📁 Batch Prediction (Folder)", "🖼 Single Image Prediction"])

# =========================================================
# TAB 1 █▀▀ Batch Prediction Using Folder Upload
# =========================================================
with tab1:
    st.subheader("📂 Upload Folder of Test Images")

    uploaded_folder = st.file_uploader(
        "Upload a folder containing images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_folder:
        st.success(f"Uploaded {len(uploaded_folder)} images.")

    if st.button("🚀 Start Folder Processing"):
        if uploaded_model_file is None:
            st.error("❌ Please upload a model file (.keras) from the sidebar.")
        elif not uploaded_folder:
            st.error("❌ Please upload a folder of images.")
        else:
            # Build dataset
            images, filenames = create_dataset_from_uploaded(uploaded_folder)
            dataset = tf.data.Dataset.from_tensor_slices(images).batch(BATCH_SIZE)

            progress_bar = st.progress(0)
            status = st.empty()

            all_preds = []
            total_batches = math.ceil(len(images) / BATCH_SIZE)

            for step, batch in enumerate(dataset):
                preds = model.predict_on_batch(batch)
                indices = np.argmax(preds, axis=1)
                all_preds.extend(indices)

                progress = (step + 1) / total_batches
                progress_bar.progress(progress)
                status.text(f"Processing {step+1}/{total_batches} batches...")

            df = pd.DataFrame({
                "ImageName": filenames,
                "ClassLabel": all_preds
            })

            st.success("✅ Processing Completed!")
            st.dataframe(df, use_container_width=True)

            # --- Download Excel ---
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            st.download_button(
                label="💾 Download Results (CSV)",
                data=buffer,
                file_name="results.csv",
                mime="text/csv"
            )

# =========================================================
# TAB 2 █▀▀ Single Image Prediction
# =========================================================
with tab2:
    st.subheader("🖼 Upload One Image")

    single_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if single_file and model:
        img = load_image_from_bytes(single_file)
        st.image(img.astype(np.uint8), caption="Uploaded Image", width=300)

        img_batch = np.expand_dims(img, axis=0)
        pred = model.predict(img_batch)
        class_id = int(np.argmax(pred))

        style_name = dp.CLASS_NAMES[class_id] if class_id < len(dp.CLASS_NAMES) else "unknown"
        st.success(f"🎯 Prediction: **{style_name}** (class {class_id})")

    elif single_file and not model:
        st.error("❌ Please upload a model first from the sidebar.")
