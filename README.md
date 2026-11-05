# PlonkNet-10 (GeoCNN-10)

Country-level geolocation of Street View images: given one street-level image, the model predicts in which of 10 countries it was taken, as in the game GeoGuessr.

**Baseline (test set):** top-1 accuracy 62.6% · top-5 accuracy 94.7% · macro F1 0.585 (chance: 10% top-1, 50% top-5).

## Countries
- **Easy group** (distinct clues according to the Plonk It guide): United States, Japan, Brazil, South Africa, Thailand
- **Hard group** (similar European countries): France, Spain, Poland, Italy, Germany

## Data
[GeoLocation – Geoguessr Images (50K)](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k) (Kaggle). After the integrity checks, the selection of 10 countries, a cap of 2,000 images per country and near-duplicate removal: 13,527 images (9,661 train / 1,933 validation / 1,933 test).

## Notebooks (run in order, in Google Colab)
| Notebook | What it does | Main outputs |
| --- | --- | --- |
| `notebooks/01_PlonkNet_DataPreparation.ipynb` | Mini ETL, data diagnosis (duplicates, imbalance, resolutions, near-duplicates), country selection, split and export | `splits.csv`, `classes.json`, `plonknet10_256.zip` |
| `notebooks/02_PlonkNet_Training.ipynb` | ResNet50 pretrained on ImageNet, frozen, with a new linear head; training on cached features; evaluation (per-country F1, confusion matrix, calibration) | `plonknet10_baseline_v1.pt`, metrics and figures |
| `notebooks/03_PlonkNet_Prediction.ipynb` | Loads the trained model and predicts the top-5 countries of any image | Predictions (console and chart), `demo_predictions_20.csv` |

The `outputs/` folder contains the small results: class list, split, metrics per epoch, final results, demo predictions and figures.

## Model weights and data
The weights, the cached features and the processed images are too large for GitHub. They are available here: https://drive.google.com/drive/folders/1cvJU1xJKnrCvmUxH2qyRhXois4tjetzW?usp=sharing

## Stack
Python · PyTorch · torchvision · scikit-learn · pandas · Google Colab

## Author
Evan André Santana Pacheco — Tecnológico de Monterrey, Campus Querétaro