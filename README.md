# Interior Design Style Classifier

A deep-learning app that predicts the design style of a room photo (17 styles such as
modern, boho, industrial, scandinavian, farmhouse...). Built with TensorFlow/Keras and Streamlit.

**Live demo:** _add your .streamlit.app link here_

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app: upload a photo, get the predicted style and top-3 probabilities |
| `data_preprocessing.py` | Class names, class weights, image preprocessing |
| `models_implementation.py` | Model definitions: simple CNN, deep CNN, custom ResNet, CCT transformer |
| `model_utils.py` | Loads saved models, including old HDF5 files that were saved with a `.keras` name |
| `convert_models.py` | Optional: converts old HDF5-style model files to real `.keras` files |
| `train.py` | Trains one model (CCT) with augmentation, class weights and callbacks |
| `train_all_models.py` | Trains and compares all four models |
| `test.py`, `ensamble_test.py` | Predict a test set with one model / an ensemble and write a submission CSV |
| `test_2.py` | Streamlit exam tool: upload a model and test images |
| `balance_data.py` | Duplicates images in small classes to balance the training folder |
| `ResNet_Custom_best.keras` | Trained model used by the app |

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Train

Place the dataset in `StyleClassificationIndoors/StyleClassificationIndoors/train/<style>/`, then:

```
pip install -r requirements-train.txt
python train_all_models.py     # all models
python train.py                # only the CCT model
```

## Model files

Models trained with TensorFlow 2.11 or older are HDF5 files even though they end in `.keras`.
`model_utils.load_model_any_format` opens both kinds. To convert them once:

```
python convert_models.py ResNet_Custom_best.keras
```
