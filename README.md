# Design of a Vision-Language Model for 3D Medical Image Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dhanuvagman006/Research_Thripthika/blob/claude/sharp-cerf-ohi7fw/notebooks/OrganVLM_Colab.ipynb)

A deep learning mini project that recognises which of 11 organs a 3D CT volume shows, comparing a baseline
3D CNN classifier with a simple vision-language model that matches each scan with a text description of the organ
(dataset: OrganMNIST3D, 1,742 volumes, 11 classes).

## Run it in Google Colab (recommended)

1. Click **Open in Colab** above (or [this link](https://colab.research.google.com/github/dhanuvagman006/Research_Thripthika/blob/claude/sharp-cerf-ohi7fw/notebooks/OrganVLM_Colab.ipynb)).
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Select **Runtime → Run all**. The full notebook takes about 10–15 minutes on a GPU.

The notebook covers the dataset exploration, architecture diagrams, training of both models and an ablation,
results (accuracy, AUC, F1, confusion matrices, ROC curves, per-class accuracy) and analysis (t-SNE embeddings,
learned filters, similarity scores, sample predictions). All figures are saved as 200-dpi PNGs.

## Repository contents

| Folder | Contents |
| --- | --- |
| `notebooks/` | `OrganVLM_Colab.ipynb`: the complete research notebook |
| `code/` | `train.py`: the same models as a command-line script (see `code/README.md`) |
| `proposal/` | Project proposal (PDF and Word); `advanced/` holds a longer, detailed version |
| `presentation/` | Narrated explainer video of the project |
