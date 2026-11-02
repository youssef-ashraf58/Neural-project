"""Helpers for loading the project's saved models.

Why this exists: older TensorFlow/Keras (<= 2.11) saved models as HDF5 even when
the file name ended in ".keras". Newer Keras only accepts real ".keras" zip files
under that name, so loading those files fails with
"File not found ... Please ensure the file is an accessible `.keras` zip file".

`load_model_any_format` detects the real format and loads either kind.
"""
import shutil
import tempfile
from pathlib import Path

HDF5_MAGIC = b"\x89HDF"


def is_hdf5_file(path):
    """True if the file is HDF5 (whatever its extension says)."""
    with open(path, "rb") as f:
        return f.read(4) == HDF5_MAGIC


def load_model_any_format(path, compile=False):
    """Load a Keras model from a real .keras zip OR an HDF5 file named .keras/.h5."""
    import tensorflow as tf

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if is_hdf5_file(path):
        # Keras picks the loader from the extension, so give it a temporary .h5 copy.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / (path.stem + ".h5")
            shutil.copyfile(path, tmp_path)
            return tf.keras.models.load_model(str(tmp_path), compile=compile)

    return tf.keras.models.load_model(str(path), compile=compile)
