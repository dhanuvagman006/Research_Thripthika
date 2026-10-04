# Organ classification on 3D CT: 3D CNN vs. a simple vision-language model

Code for the project proposal *Design of a Vision-Language Model for 3D Medical Image Analysis*
(see `../proposal/`). It trains and compares two models on the OrganMNIST3D dataset
(1,742 CT volumes of 28 × 28 × 28 voxels, 11 organ classes).

| Model | What it does |
| --- | --- |
| `baseline` | 3D CNN (four conv blocks, 32–256 filters) + fully connected layer, trained with cross-entropy |
| `vlm` | Same 3D CNN + a projection layer. Each scan is matched against frozen ClinicalBERT vectors of the 11 sentences "A CT scan of the &lt;organ&gt;." and trained with the SigLIP loss. The best-matching sentence is the prediction |

## Setup

```bash
pip install -r requirements.txt
```

The dataset (33 MB) downloads automatically on the first run. The `vlm` model also downloads
ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`, about 440 MB) the first time.

## Run

```bash
python train.py --model baseline
python train.py --model vlm
```

Options: `--epochs` (default 50), `--batch-size` (32), `--lr` (0.001), `--data-dir` (`./data`), `--seed` (0).
A GPU is used automatically if available; otherwise the script runs on the CPU.

Each run prints the validation accuracy per epoch, then the test accuracy, AUC and confusion matrix of the
model with the best validation accuracy, and saves its weights as `<model>_best.pt`.
Rows and columns of the confusion matrix follow the class order: liver, right kidney, left kidney,
right femur, left femur, bladder, heart, right lung, left lung, spleen, pancreas.

**Google Colab:** upload `train.py` (or clone this repository), then run
`!pip install medmnist` and `!python train.py --model vlm`. PyTorch, Transformers and scikit-learn are
already installed on Colab.

## Results

Test set (610 volumes), default settings (50 epochs, batch size 32, learning rate 0.001, seed 0),
4-core CPU without a GPU:

| Model | Test accuracy | AUC | Best validation accuracy | Trainable parameters | Training time |
| --- | --- | --- | --- | --- | --- |
| `baseline` | 80.5% | 0.981 | 88.2% | 1,166,347 | 7.2 min |
| `vlm` | 80.0% | 0.979 | 90.7% | 1,360,898 | 6.9 min |

Results from a single run; expect a few points of variation between seeds and hardware.

## Note on the text vectors

The 11 organ sentences differ by only one word, so their raw ClinicalBERT vectors are almost identical
(average cosine similarity 0.985). Without a fix, the `vlm` model reached only 66.6% test accuracy after
30 epochs, against 73.9% for the baseline. `train.py` therefore subtracts the mean of the 11 vectors
(centring), which makes them distinct (average cosine −0.10) and brings the `vlm` model level with the
baseline.
