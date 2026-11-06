# PlonkNet-10 (GeoCNN-10)

Country-level geolocation of Street View images: given one street-level image, the model predicts in which of 10 countries it was taken, as in the game GeoGuessr.

**Baseline (test set):** top-1 accuracy 62.6% · top-5 accuracy 94.7% · macro F1 0.585 (chance: 10% top-1, 50% top-5).

## Try it with your own image
The prediction notebook already has a cell to upload images and see the model's answer.

1. Open `notebooks/03_PlonkNet_Prediction.ipynb` in Google Colab: [open in Colab](https://colab.research.google.com/github/AelAndre/PlonkNet-10/blob/main/notebooks/03_PlonkNet_Prediction.ipynb).
2. Get the trained weights: download `plonknet10_baseline_v1.pt` from the [Drive folder](https://drive.google.com/drive/folders/1cvJU1xJKnrCvmUxH2qyRhXois4tjetzW?usp=sharing), upload it to your own Google Drive and change `CHECKPOINT_PATH` in the setup cell to its location. This is the only line you need to change.
3. Run the cells in order: Setup, Load the model, and Preprocessing.
4. In the section **Try it: upload your images here**, run the cell and choose one or more images (a GeoGuessr screenshot or any street-level photo).

For each image, the notebook shows the image as the model sees it (center-cropped to the panoramic ratio used in training) and the 5 most likely countries, as a chart and as text in the console. A GPU is not needed: the default Colab CPU is enough.

Keep in mind that the model only knows the 10 countries below: an image from any other country will still be assigned to one of them.

## Countries
- **Easy group** (distinct clues according to the Plonk It guide): United States, Japan, Brazil, South Africa, Thailand
- **Hard group** (similar European countries): France, Spain, Poland, Italy, Germany

## Data
[GeoLocation – Geoguessr Images (50K)](https://www.kaggle.com/datasets/ubitquitin/geolocation-geoguessr-images-50k) (Kaggle). After the integrity checks, the selection of 10 countries, a cap of 2,000 images per country and near-duplicate removal: 13,527 images (9,661 train / 1,933 validation / 1,933 test).

## Model
ResNet50 pretrained on ImageNet, frozen, with a new linear head of 10 outputs (20,490 trainable parameters out of 23.5 million). The input is 224 × 512 to keep the panoramic shape of the screenshots. Because the backbone is frozen, the 2,048 features of each image were computed once and cached, so the head trains in seconds, even without a GPU.

## Notebooks (run in order, in Google Colab)
| Notebook | What it does | Main outputs |
| --- | --- | --- |
| `notebooks/01_PlonkNet_DataPreparation.ipynb` | Mini ETL, data diagnosis (duplicates, imbalance, resolutions, near-duplicates), country selection, split and export | `splits.csv`, `classes.json`, `plonknet10_256.zip` |
| `notebooks/02_PlonkNet_Training.ipynb` | ResNet50 pretrained on ImageNet, frozen, with a new linear head; training on cached features; evaluation (per-country F1, confusion matrix, calibration) | `plonknet10_baseline_v1.pt`, metrics and figures |
| `notebooks/03_PlonkNet_Prediction.ipynb` | Loads the trained model and predicts the top-5 countries of any image you upload; demo with 20 random Street View images | Predictions (console and chart), `demo_predictions_20.csv` |

The `outputs/` folder contains the small results: class list, split, metrics per epoch, final results, demo predictions and figures.

## Report
`report/PlonkNet10_report.pdf` (IEEE conference format) and its LaTeX source.

## Model weights and data
The weights, the cached features and the processed images are too large for GitHub. They are available here: https://drive.google.com/drive/folders/1cvJU1xJKnrCvmUxH2qyRhXois4tjetzW?usp=sharing

## Next steps
The evaluation of the baseline points to the next adjustments, each tested one at a time on the validation set:
- more epochs, since the head had not converged after 10;
- an MLP head (2048 → 512 → 10 with dropout), since a single linear layer reaches only 69% on the training set;
- class weights, since the model over-predicts the larger classes and almost never predicts Italy;
- temperature scaling, since the model is underconfident.

Later, with a GPU: fine-tuning the last block of the ResNet and augmentation without horizontal flips.

## Stack
Python · PyTorch · torchvision · scikit-learn · pandas · Google Colab

## Author
Evan André Santana Pacheco — Tecnológico de Monterrey, Campus Querétaro
