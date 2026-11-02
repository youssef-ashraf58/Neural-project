"""Converts old HDF5-style model files to real .keras files.

Usage (from the project folder):
    python convert_models.py ResNet_Custom_best.keras
    python convert_models.py            # converts every *.keras file in the folder

Each model is written next to the original as <name>_converted.keras.
Originals are never modified. Rename the converted file if you want to use it.
"""
import sys
import zipfile
from pathlib import Path

from model_utils import is_hdf5_file, load_model_any_format


def convert(path):
    path = Path(path)
    if zipfile.is_zipfile(path):
        print(f"skip   {path.name}: already a valid .keras file")
        return
    if not is_hdf5_file(path):
        print(f"skip   {path.name}: not an HDF5 or .keras file")
        return
    out = path.with_name(path.stem + "_converted.keras")
    try:
        model = load_model_any_format(path, compile=False)
        model.save(out)
        print(f"ok     {path.name} -> {out.name}")
    except Exception as e:  # e.g. custom layers that need to be registered
        print(f"FAILED {path.name}: {e}")


if __name__ == "__main__":
    targets = [Path(p) for p in sys.argv[1:]] or sorted(
        p for p in Path(".").glob("*.keras") if not p.stem.endswith("_converted")
    )
    for target in targets:
        convert(target)
