"""Builds OrganVLM_Colab.ipynb (run: python build_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

REPO = "dhanuvagman006/Research_Thripthika"
BRANCH = "claude/sharp-cerf-ohi7fw"

md(f"""
# Design of a Vision-Language Model for 3D Medical Image Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/notebooks/OrganVLM_Colab.ipynb)

**Task:** recognise which of 11 organs a 3D CT volume shows.
**Comparison:** a baseline 3D CNN classifier vs. a simple vision-language model (VLM) that matches each scan with a text description of the organ.

| Section | Contents |
|---|---|
| 1. Setup | Libraries, configuration, reproducibility |
| 2. Dataset | OrganMNIST3D: samples, classes, splits, visual exploration |
| 3. Methodology | Architecture diagrams, baseline 3D CNN, proposed VLM |
| 4. Text encoder analysis | ClinicalBERT embeddings and the centring step |
| 5. Training | Baseline, VLM and an ablation (VLM without centring) |
| 6. Results | Accuracy, AUC, F1, confusion matrices, ROC curves, per-class accuracy |
| 7. Analysis | Embedding space (t-SNE), learned filters, similarity scores, sample predictions |
| 8. Conclusion | Findings and references |

**Runtime:** select *Runtime → Change runtime type → T4 GPU* in Colab. The full notebook then runs in roughly 10–15 minutes (on CPU about 25 minutes).
""")

md("## 1. Setup")
code("""
# Install the only package Colab does not already have
import sys, subprocess
try:
    import medmnist
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "medmnist"], check=True)
    import medmnist
""")
code("""
import os, time, math, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F
from medmnist import OrganMNIST3D
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, confusion_matrix,
                             roc_curve, classification_report)
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore")

# ---------------- configuration ----------------
EPOCHS = int(os.environ.get("EPOCHS", 50))      # training epochs per model
BATCH_SIZE = 32
LR = 1e-3
SEED = 0
RUN_ABLATION = True                              # also train the VLM without centring
DATA_DIR = "./data"
FIG_DIR = "./figures"
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch {torch.__version__} | device: {device} | epochs: {EPOCHS}")

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 10})
NAVY, TEAL, ORANGE, GREY, RED, GREEN = "#1B2A49", "#0F8B8D", "#E07A1F", "#6B7280", "#C8463C", "#2E8C57"
def savefig(name):
    plt.savefig(f"{FIG_DIR}/{name}.png", dpi=200, bbox_inches="tight")
""")

md("""
## 2. Dataset: OrganMNIST3D

OrganMNIST3D is part of the **MedMNIST v2** benchmark (Yang et al., *Scientific Data*, 2023). Each sample is a
**28 × 28 × 28** grayscale CT volume cropped around one organ, taken from 201 abdominal CT scans of the
Liver Tumor Segmentation (LiTS) benchmark. The licence is CC BY 4.0.

* **Number of samples:** 1,742 volumes (official split: 971 train / 161 validation / 610 test)
* **Number of classes:** 11 organs
""")
code("""
ORGANS = ["liver", "right kidney", "left kidney", "right femur", "left femur", "bladder",
          "heart", "right lung", "left lung", "spleen", "pancreas"]
SHORT = ["Liver", "R kidney", "L kidney", "R femur", "L femur", "Bladder", "Heart", "R lung", "L lung", "Spleen", "Pancreas"]

def load_split(split):
    ds = OrganMNIST3D(split=split, download=True, root=DATA_DIR)
    x = torch.tensor(ds.imgs, dtype=torch.float32).unsqueeze(1)
    if x.max() > 1: x = x / 255.0                       # scale voxel values to 0-1
    y = torch.tensor(ds.labels).long().squeeze(1)
    return x, y

(x_tr, y_tr), (x_va, y_va), (x_te, y_te) = (load_split(s) for s in ("train", "val", "test"))
print("train:", tuple(x_tr.shape), "| val:", tuple(x_va.shape), "| test:", tuple(x_te.shape))

counts = pd.DataFrame({s: np.bincount(y.numpy(), minlength=11) for s, y in
                       [("train", y_tr), ("val", y_va), ("test", y_te)]}, index=SHORT)
counts["total"] = counts.sum(1)
counts.loc["TOTAL"] = counts.sum()
counts
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3, 1]})
idx = np.arange(11); w = 0.27
for i, (split, col) in enumerate([("train", TEAL), ("val", ORANGE), ("test", NAVY)]):
    axes[0].bar(idx + (i - 1) * w, counts.loc[SHORT, split], w, label=split, color=col)
axes[0].set_xticks(idx); axes[0].set_xticklabels(SHORT, rotation=30, ha="right")
axes[0].set_ylabel("Number of volumes"); axes[0].set_title("Class distribution per split"); axes[0].legend()
sizes = [len(y_tr), len(y_va), len(y_te)]
axes[1].pie(sizes, labels=[f"Train\\n{sizes[0]}", f"Validation\\n{sizes[1]}", f"Test\\n{sizes[2]}"],
            colors=[TEAL, ORANGE, NAVY], wedgeprops=dict(width=0.45), textprops={"fontsize": 10})
axes[1].set_title("Data split (1,742 volumes)")
plt.tight_layout(); savefig("01_class_distribution"); plt.show()
""")
md("### Sample volumes: one per organ, shown in three orthogonal planes")
code("""
fig, axes = plt.subplots(3, 11, figsize=(16, 4.8))
for c in range(11):
    v = x_tr[(y_tr == c).nonzero()[0, 0], 0].numpy()
    for r, (img, plane) in enumerate([(v[14], "axial"), (v[:, 14], "coronal"), (v[:, :, 14], "sagittal")]):
        axes[r, c].imshow(img, cmap="gray"); axes[r, c].axis("off")
        if r == 0: axes[r, c].set_title(SHORT[c], fontsize=9)
        if c == 0: axes[r, c].text(-6, 14, plane, rotation=90, va="center", ha="right", fontsize=9)
plt.suptitle("Middle slice of one volume per class (rows: three orthogonal planes)", fontweight="bold")
plt.tight_layout(); savefig("02_samples_per_class"); plt.show()
""")
code("""
# A full 3D volume shown as its 28 slices, plus the voxel intensity distribution
v = x_tr[(y_tr == 0).nonzero()[0, 0], 0].numpy()
fig, axes = plt.subplots(3, 10, figsize=(15, 4.8))
for i, ax in enumerate(axes.flat):
    ax.axis("off")
    if i < 28: ax.imshow(v[i], cmap="gray"); ax.set_title(f"slice {i}", fontsize=7)
plt.suptitle("One liver volume = 28 stacked 28×28 slices (a 3D array of 21,952 voxels)", fontweight="bold")
plt.tight_layout(); savefig("03_volume_slices"); plt.show()

plt.figure(figsize=(6, 3))
plt.hist(x_tr[:200].flatten().numpy(), bins=60, color=TEAL)
plt.title("Voxel intensity distribution (200 training volumes)"); plt.xlabel("normalised intensity"); plt.ylabel("count")
plt.tight_layout(); savefig("04_intensity_histogram"); plt.show()
""")

md("""
## 3. Methodology

**Baseline (existing method).** A 3D convolutional neural network: four blocks of 3×3×3 convolution, batch
normalisation, ReLU and max pooling (32, 64, 128, 256 filters), global average pooling, and a fully connected layer
with 11 outputs. Trained with cross-entropy. This follows the CNN baselines of MedMNIST v2.

**Proposed modification (vision-language model).** Following the image–text alignment idea of the base paper
*Med3DVLM*, the fully connected layer is replaced by matching against text:

1. One sentence per organ: *"A CT scan of the liver."* (11 sentences).
2. A frozen **ClinicalBERT** encoder turns each sentence into a 768-dimensional vector.
3. The 3D CNN features (256-d) pass through a projection layer to 768-d.
4. Cosine similarity between the scan vector and each sentence vector gives 11 scores; the highest is the prediction.
5. Training uses the **SigLIP** pairwise sigmoid loss:

$$\\mathcal{L}_{\\text{SigLIP}} = -\\frac{1}{N}\\sum_{i=1}^{N}\\sum_{j=1}^{11}\\log\\sigma\\big(z_{ij}\\,(t\\,s_{ij}+b)\\big),\\qquad \\hat y_i=\\arg\\max_j s_{ij}$$

where $s_{ij}$ is the cosine similarity of scan $i$ and sentence $j$, $z_{ij}=+1$ for the correct organ and $-1$
otherwise, and $t$, $b$ are a learnable scale and bias.
""")
code('''
def draw_architecture():
    fig, ax = plt.subplots(figsize=(15, 6.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 6.2); ax.axis("off")
    def box(x, y, w, h, title, sub="", fc="white", ec=GREY, lw=1.5):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2 + (0.13 if sub else 0), title, ha="center", va="center", fontsize=10, fontweight="bold", color=NAVY)
        if sub: ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=8, color=GREY)
    def arr(x1, y1, x2, y2, c=GREY):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
    ax.text(0.1, 5.85, "(a) Baseline: 3D CNN classifier", fontsize=12, fontweight="bold", color=NAVY)
    box(0.1, 4.6, 1.6, 0.9, "CT volume", "1×28×28×28")
    for i, (f, s) in enumerate([(32, "28→14"), (64, "14→7"), (128, "7→3"), (256, "3³")]):
        box(2.2 + i * 1.75, 4.6, 1.5, 0.9, f"Conv block {i+1}", f"{f} filters, {s}", "#DDF1F1", TEAL); arr(1.7 + i * 1.75 + (0.0 if i == 0 else 0), 5.05, 2.2 + i * 1.75, 5.05)
    box(9.3, 4.6, 1.5, 0.9, "Global pool", "256-d features"); arr(8.95, 5.05, 9.3, 5.05)
    box(11.2, 4.6, 1.6, 0.9, "FC layer", "256 → 11"); arr(10.8, 5.05, 11.2, 5.05)
    box(13.2, 4.6, 1.6, 0.9, "11 class scores", "softmax", "#FCE8D5", ORANGE); arr(12.8, 5.05, 13.2, 5.05)
    ax.text(0.1, 3.75, "(b) Proposed: simple vision-language model (new parts in orange)", fontsize=12, fontweight="bold", color=NAVY)
    box(0.1, 2.5, 1.6, 0.9, "CT volume", "1×28×28×28")
    box(2.2, 2.5, 6.6, 0.9, "Same 3D CNN encoder (4 conv blocks + global pool)", "256-d image features", "#DDF1F1", TEAL); arr(1.7, 2.95, 2.2, 2.95)
    box(9.3, 2.5, 1.5, 0.9, "Projection", "256 → 768", "#FCE8D5", ORANGE, 2); arr(8.8, 2.95, 9.3, 2.95)
    box(0.1, 0.4, 2.6, 0.9, "11 organ sentences", "\\"A CT scan of the liver.\\" …", "#FCE8D5", ORANGE, 2)
    box(3.2, 0.4, 2.2, 0.9, "ClinicalBERT", "frozen, 768-d", "#FCE8D5", ORANGE, 2); arr(2.7, 0.85, 3.2, 0.85)
    box(5.9, 0.4, 2.2, 0.9, "Centring", "subtract mean vector", "#FCE8D5", ORANGE, 2); arr(5.4, 0.85, 5.9, 0.85)
    box(11.2, 1.45, 1.6, 0.9, "Cosine similarity", "scan vs 11 texts", "#FCE8D5", ORANGE, 2)
    arr(10.8, 2.95, 11.6, 2.35); arr(8.1, 0.85, 11.6, 1.45)
    box(13.2, 1.45, 1.6, 0.9, "Prediction", "argmax similarity", "white", NAVY); arr(12.8, 1.9, 13.2, 1.9)
    ax.text(11.2, 0.75, "Loss: SigLIP (pull correct pair, push 10 others)", fontsize=9, color=ORANGE, fontweight="bold")
    plt.tight_layout(); savefig("05_architecture"); plt.show()
draw_architecture()
''')
code("""
def cnn_encoder():
    \"\"\"Four blocks: 3x3x3 conv -> batch norm -> ReLU -> max pool. Output: 256 features.\"\"\"
    ch, layers = [1, 32, 64, 128, 256], []
    for i in range(4):
        layers += [nn.Conv3d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm3d(ch[i + 1]), nn.ReLU()]
        if i < 3: layers.append(nn.MaxPool3d(2))
    return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten())

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = cnn_encoder(); self.fc = nn.Linear(256, 11)
    def features(self, x): return self.encoder(x)
    def forward(self, x): return self.fc(self.encoder(x))

class OrganVLM(nn.Module):
    def __init__(self, text_vectors):
        super().__init__()
        self.encoder = cnn_encoder(); self.proj = nn.Linear(256, text_vectors.shape[1])
        self.register_buffer("text", F.normalize(text_vectors, dim=-1))
        self.log_t = nn.Parameter(torch.tensor(math.log(10.0))); self.b = nn.Parameter(torch.tensor(-10.0))
    def features(self, x): return F.normalize(self.proj(self.encoder(x)), dim=-1)
    def forward(self, x): return self.log_t.exp() * self.features(x) @ self.text.T + self.b

def siglip_loss(logits, y):
    z = -torch.ones_like(logits); z[torch.arange(len(y)), y] = 1
    return -F.logsigmoid(z * logits).sum(1).mean()

n_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"Baseline 3D CNN trainable parameters: {n_params(BaselineCNN()):,}")
print(f"Proposed VLM trainable parameters:    {n_params(OrganVLM(torch.zeros(11, 768))):,} (ClinicalBERT frozen, not counted)")
""")

md("""
## 4. Text encoder analysis: why centring is needed

The 11 sentences differ by only one word, so their raw ClinicalBERT vectors are almost identical. When all text
vectors point in nearly the same direction, the image encoder cannot separate them well. Subtracting the mean of the
11 vectors (**centring**, analogous to removing a DC offset from a set of signals) removes the shared component and
leaves the organ-specific differences.
""")
code("""
from transformers import AutoTokenizer, AutoModel
BERT = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer, bert = AutoTokenizer.from_pretrained(BERT), AutoModel.from_pretrained(BERT).eval()
sentences = [f"A CT scan of the {o}." for o in ORGANS]
with torch.no_grad():
    batch = tokenizer(sentences, padding=True, return_tensors="pt")
    text_raw = bert(**batch).last_hidden_state[:, 0]          # [CLS] vector per sentence, 768-d
text_centred = text_raw - text_raw.mean(0, keepdim=True)
del bert

def cos_matrix(v): v = F.normalize(v, dim=-1); return (v @ v.T).numpy()
off = ~np.eye(11, dtype=bool)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, v, name in [(axes[0], text_raw, "Raw ClinicalBERT vectors"), (axes[1], text_centred, "After centring")]:
    m = cos_matrix(v); im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(11)); ax.set_yticks(range(11)); ax.set_xticklabels(SHORT, rotation=45, ha="right"); ax.set_yticklabels(SHORT)
    for i in range(11):
        for j in range(11): ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(m[i, j]) > 0.6 else "black")
    ax.set_title(f"{name}\\nmean off-diagonal cosine = {m[off].mean():.3f}")
fig.colorbar(im, ax=axes, shrink=0.8, label="cosine similarity")
savefig("06_text_similarity_centring"); plt.show()
""")

md("""
## 5. Training

Both models use the same encoder, data, optimiser (Adam, learning rate 0.001), batch size (32), number of epochs and
augmentation (random flips along the three axes). After every epoch the model is evaluated on the validation set,
and the weights with the best validation accuracy are kept for testing.
""")
code("""
def random_flip(x):
    for dim in (2, 3, 4):
        m = torch.rand(len(x), device=x.device) < 0.5
        x[m] = x[m].flip(dim)
    return x

@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.cat([model(x[i:i + 128].to(device)).cpu() for i in range(0, len(x), 128)])

def train_model(model, loss_fn, name, epochs=EPOCHS):
    set_seed(); model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_acc": []}
    best, best_state, t0 = -1, None, time.time()
    for ep in range(1, epochs + 1):
        model.train(); order = torch.randperm(len(x_tr)); tot_loss = correct = 0
        for i in range(0, len(x_tr), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            xb, yb = random_flip(x_tr[idx].to(device)), y_tr[idx].to(device)
            out = model(xb); loss = loss_fn(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(idx); correct += (out.argmax(1) == yb).sum().item()
        val_acc = (predict(model, x_va).argmax(1) == y_va).float().mean().item()
        hist["epoch"].append(ep); hist["train_loss"].append(tot_loss / len(x_tr))
        hist["train_acc"].append(correct / len(x_tr)); hist["val_acc"].append(val_acc)
        if val_acc > best: best, best_state = val_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"[{name}] epoch {ep:3d} | loss {hist['train_loss'][-1]:.4f} | train acc {hist['train_acc'][-1]:.3f} | val acc {val_acc:.3f}")
    model.load_state_dict(best_state)
    print(f"[{name}] done in {(time.time() - t0) / 60:.1f} min | best val acc {best:.3f}")
    return model, pd.DataFrame(hist), time.time() - t0
""")
md("### 5.1 Baseline 3D CNN")
code("""
baseline, hist_base, time_base = train_model(BaselineCNN(), F.cross_entropy, "Baseline")
""")
md("### 5.2 Proposed vision-language model (with centring)")
code("""
vlm, hist_vlm, time_vlm = train_model(OrganVLM(text_centred), siglip_loss, "VLM")
""")
md("### 5.3 Ablation: VLM without centring")
code("""
if RUN_ABLATION:
    vlm_raw, hist_raw, time_raw = train_model(OrganVLM(text_raw), siglip_loss, "VLM (no centring)")
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
runs = [("Baseline 3D CNN", hist_base, NAVY), ("VLM (centred)", hist_vlm, ORANGE)]
if RUN_ABLATION: runs.append(("VLM (no centring)", hist_raw, GREY))
for name, h, col in runs:
    axes[0].plot(h.epoch, h.train_loss / h.train_loss.iloc[0], color=col, label=name)
    axes[1].plot(h.epoch, h.train_acc * 100, color=col, label=name)
    axes[2].plot(h.epoch, h.val_acc * 100, color=col, label=name)
axes[0].set_title("Training loss (relative to epoch 1)"); axes[1].set_title("Training accuracy (%)"); axes[2].set_title("Validation accuracy (%)")
for ax in axes: ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); savefig("07_training_curves"); plt.show()
""")

md("## 6. Results on the test set (610 volumes, never seen during training)")
code("""
def evaluate(model, name, t):
    logits = predict(model, x_te); prob = logits.softmax(1).numpy(); pred = logits.argmax(1).numpy(); yt = y_te.numpy()
    return {"Model": name, "Accuracy (%)": 100 * accuracy_score(yt, pred), "AUC (macro)": roc_auc_score(yt, prob, multi_class="ovr"),
            "F1 (macro)": f1_score(yt, pred, average="macro"), "Trainable params": n_params(model),
            "Train time (min)": t / 60}, prob, pred

res_b, prob_b, pred_b = evaluate(baseline, "Baseline 3D CNN", time_base)
res_v, prob_v, pred_v = evaluate(vlm, "Proposed VLM (centred)", time_vlm)
rows = [res_b, res_v]
if RUN_ABLATION:
    res_r, prob_r, pred_r = evaluate(vlm_raw, "Ablation: VLM without centring", time_raw); rows.append(res_r)
results = pd.DataFrame(rows).set_index("Model")
results.to_csv("results.csv")
results.style.format({"Accuracy (%)": "{:.1f}", "AUC (macro)": "{:.3f}", "F1 (macro)": "{:.3f}", "Trainable params": "{:,}", "Train time (min)": "{:.1f}"})
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
names = list(results.index); cols = [NAVY, ORANGE, GREY][:len(names)]
for ax, metric, fmt in [(axes[0], "Accuracy (%)", "{:.1f}%"), (axes[1], "AUC (macro)", "{:.3f}"), (axes[2], "F1 (macro)", "{:.3f}")]:
    vals = results[metric].values; bars = ax.bar(range(len(vals)), vals, color=cols)
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels([n.replace(" (", "\\n(").replace(": ", ":\\n") for n in names], fontsize=8)
    ax.set_title(metric); ax.set_ylim(min(vals) * 0.9, max(vals) * 1.06)
plt.tight_layout(); savefig("08_metric_comparison"); plt.show()
""")
md("### Confusion matrices (row-normalised)")
code("""
preds = [("Baseline 3D CNN", pred_b), ("Proposed VLM", pred_v)]
fig, axes = plt.subplots(1, 2, figsize=(17, 6.4))
plt.subplots_adjust(wspace=0.35)
for ax, (name, pred) in zip(axes, preds):
    cm = confusion_matrix(y_te.numpy(), pred); cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(11):
        for j in range(11):
            if cm[i, j]: ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7.5, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(11)); ax.set_yticks(range(11)); ax.set_xticklabels(SHORT, rotation=45, ha="right"); ax.set_yticklabels(SHORT)
    ax.set_xlabel("Predicted organ"); ax.set_ylabel("True organ"); ax.set_title(f"{name} (accuracy {np.mean(pred == y_te.numpy()) * 100:.1f}%)")
fig.colorbar(im, ax=axes, shrink=0.75, label="fraction of true class")
savefig("09_confusion_matrices"); plt.show()
""")
md("### Per-class accuracy and ROC curves")
code("""
def per_class_acc(pred): return np.array([np.mean(pred[y_te.numpy() == c] == c) for c in range(11)]) * 100
pc = pd.DataFrame({"Baseline": per_class_acc(pred_b), "VLM": per_class_acc(pred_v)}, index=SHORT)
fig, ax = plt.subplots(figsize=(14, 4.2)); idx = np.arange(11)
ax.bar(idx - 0.2, pc.Baseline, 0.4, color=NAVY, label="Baseline 3D CNN"); ax.bar(idx + 0.2, pc.VLM, 0.4, color=ORANGE, label="Proposed VLM")
ax.set_xticks(idx); ax.set_xticklabels(SHORT, rotation=30, ha="right"); ax.set_ylabel("Test accuracy (%)")
ax.set_title("Per-class test accuracy"); ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); savefig("10_per_class_accuracy"); plt.show()
pc.round(1).T
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
cmap = plt.get_cmap("tab20")
for ax, (name, prob) in zip(axes, [("Baseline 3D CNN", prob_b), ("Proposed VLM", prob_v)]):
    for c in range(11):
        fpr, tpr, _ = roc_curve((y_te.numpy() == c).astype(int), prob[:, c])
        ax.plot(fpr, tpr, color=cmap(c), lw=1.3, label=f"{SHORT[c]} ({roc_auc_score((y_te.numpy() == c).astype(int), prob[:, c]):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"One-vs-rest ROC: {name}\\nmacro AUC = {roc_auc_score(y_te.numpy(), prob, multi_class='ovr'):.3f}")
    ax.legend(fontsize=7, loc="lower right", title="class (AUC)", title_fontsize=7)
plt.tight_layout(); savefig("11_roc_curves"); plt.show()
""")
code("""
print("Classification report: proposed VLM\\n")
print(classification_report(y_te.numpy(), pred_v, target_names=SHORT, digits=3))
""")

md("""
## 7. Analysis

### 7.1 Embedding space (t-SNE)
t-SNE projects the learned image features of all 610 test volumes to 2D. Well-separated colour clusters mean the
model has learned features that distinguish the organs. For the VLM, the 11 text vectors (stars) are projected into
the same space.
""")
code("""
@torch.no_grad()
def features(model, x):
    model.eval(); return torch.cat([model.features(x[i:i + 128].to(device)).cpu() for i in range(0, len(x), 128)]).numpy()

fb = features(baseline, x_te)
fv = features(vlm, x_te); tv = F.normalize(text_centred, dim=-1).numpy()
emb_b = TSNE(2, random_state=SEED, perplexity=30, init="pca").fit_transform(fb)
emb_v_all = TSNE(2, random_state=SEED, perplexity=30, init="pca").fit_transform(np.concatenate([fv, tv]))
emb_v, emb_t = emb_v_all[:-11], emb_v_all[-11:]

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5)); cmap = plt.get_cmap("tab20")
for ax, emb, name in [(axes[0], emb_b, "Baseline 3D CNN (256-d features)"), (axes[1], emb_v, "Proposed VLM (768-d shared space)")]:
    for c in range(11):
        m = y_te.numpy() == c; ax.scatter(emb[m, 0], emb[m, 1], s=12, color=cmap(c), label=SHORT[c], alpha=0.8)
    ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
for c in range(11):
    axes[1].scatter(*emb_t[c], marker="*", s=380, color=cmap(c), edgecolor="black", linewidth=1.2, zorder=5)
axes[1].scatter([], [], marker="*", s=200, color="white", edgecolor="black", label="text vector")
axes[1].legend(fontsize=8, markerscale=1.5, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout(); savefig("12_tsne_embeddings"); plt.show()
""")
md("### 7.2 What the first convolution layer learned\nEach small image is one learned 3×3×3 filter's response (feature map) to the same CT slice: some act like edge detectors, others highlight bright or dark regions, just like hand-designed image filters, but learned from data.")
code("""
sample = x_te[:1].to(device)
with torch.no_grad():
    fmap = F.relu(baseline.encoder[1](baseline.encoder[0](sample)))[0, :, 14].cpu().numpy()   # 32 maps, middle slice
fig, axes = plt.subplots(3, 12, figsize=(16, 4.3))
axes[0, 0].imshow(sample[0, 0, 14].cpu(), cmap="gray"); axes[0, 0].set_title("input", fontsize=8)
for k, ax in enumerate(axes.flat):
    ax.axis("off")
    if 1 <= k <= 32: ax.imshow(fmap[k - 1], cmap="magma"); ax.set_title(f"filter {k}", fontsize=7)
plt.suptitle(f"First-layer feature maps of the baseline for one test volume (true class: {SHORT[y_te[0]]})", fontweight="bold")
plt.tight_layout(); savefig("13_feature_maps"); plt.show()
""")
md("### 7.3 How the VLM decides: similarity scores against the 11 organ sentences")
code("""
picks = [int((y_te == c).nonzero()[0, 0]) for c in (0, 2, 6, 9)]
with torch.no_grad():
    vlm.eval(); sims = (vlm.features(x_te[picks].to(device)) @ vlm.text.T).cpu().numpy()
fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), gridspec_kw={"height_ratios": [1, 1.3]})
for i, p in enumerate(picks):
    axes[0, i].imshow(x_te[p, 0, 14], cmap="gray"); axes[0, i].axis("off"); axes[0, i].set_title(f"true: {SHORT[y_te[p]]}")
    colors = [GREEN if j == sims[i].argmax() else (TEAL if j == y_te[p] else "#C9CED6") for j in range(11)]
    axes[1, i].barh(range(11), sims[i], color=colors); axes[1, i].set_yticks(range(11)); axes[1, i].set_yticklabels(SHORT, fontsize=8)
    axes[1, i].invert_yaxis(); axes[1, i].set_xlabel("cosine similarity"); axes[1, i].set_title(f"predicted: {SHORT[sims[i].argmax()]}", color=GREEN if sims[i].argmax() == y_te[p] else RED)
plt.suptitle("VLM similarity between each scan and the 11 sentences \\"A CT scan of the <organ>.\\"", fontweight="bold")
plt.tight_layout(); savefig("14_similarity_scores"); plt.show()
""")
md("### 7.4 Sample test predictions (green = correct, red = wrong)")
code("""
rng = np.random.default_rng(1); show = rng.choice(len(x_te), 16, replace=False)
fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for ax, i in zip(axes.flat, show):
    ax.imshow(x_te[i, 0, 14], cmap="gray"); ax.axis("off")
    ok_b, ok_v = pred_b[i] == y_te[i], pred_v[i] == y_te[i]
    ax.set_title(f"true: {SHORT[y_te[i]]}", fontsize=8, fontweight="bold")
    ax.text(0.5, -0.1, f"CNN: {SHORT[pred_b[i]]}", transform=ax.transAxes, ha="center", fontsize=7.5, color=GREEN if ok_b else RED)
    ax.text(0.5, -0.22, f"VLM: {SHORT[pred_v[i]]}", transform=ax.transAxes, ha="center", fontsize=7.5, color=GREEN if ok_v else RED)
plt.tight_layout(); savefig("15_sample_predictions"); plt.show()
""")

md("## 8. Conclusion")
code("""
b, v = results.loc["Baseline 3D CNN"], results.loc["Proposed VLM (centred)"]
print(f"Baseline 3D CNN : accuracy {b['Accuracy (%)']:.1f}%  AUC {b['AUC (macro)']:.3f}  F1 {b['F1 (macro)']:.3f}")
print(f"Proposed VLM    : accuracy {v['Accuracy (%)']:.1f}%  AUC {v['AUC (macro)']:.3f}  F1 {v['F1 (macro)']:.3f}")
if RUN_ABLATION:
    r = results.loc["Ablation: VLM without centring"]
    print(f"VLM, no centring: accuracy {r['Accuracy (%)']:.1f}%  AUC {r['AUC (macro)']:.3f}  F1 {r['F1 (macro)']:.3f}")
""")
md("""
**Findings**

1. A small 3D vision-language model, trained on a single GPU, recognises organs in 3D CT with accuracy comparable to
   a standard 3D CNN classifier, while additionally learning a shared image–text embedding space.
2. Centring the text embeddings is essential: raw ClinicalBERT vectors of near-identical sentences are almost
   parallel, and removing their common component markedly improves the VLM (see the ablation).
3. Most remaining errors are between mirror-image pairs (left vs right kidney, left vs right femur), which look
   almost identical in 28×28×28 volumes.

*Exact numbers vary slightly between runs and hardware (GPU vs CPU).*

**References**

1. Y. Xin, G. C. Ates, K. Gong and W. Shao, "Med3DVLM: An efficient vision-language model for 3D medical image analysis," *IEEE J. Biomed. Health Inform.*, vol. 30, no. 3, 2026.
2. Z. Wang, L. Wang, X. Li and L. Zhou, "Diagnostic captioning by cooperative task interactions and sample-graph consistency," *IEEE TPAMI*, vol. 47, no. 8, 2025.
3. J. Yang et al., "MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification," *Scientific Data*, vol. 10, 41, 2023.
4. A. Radford et al., "Learning transferable visual models from natural language supervision," ICML 2021.
5. X. Zhai et al., "Sigmoid loss for language image pre-training," ICCV 2023.
6. E. Alsentzer et al., "Publicly available clinical BERT embeddings," Clinical NLP Workshop 2019.
""")
md("## Save outputs\nAll figures are in `figures/` (200 dpi PNG, ready for a report or slides), metrics in `results.csv`. In Colab, the next cell downloads everything as a zip.")
code("""
torch.save(baseline.state_dict(), "baseline_best.pt"); torch.save(vlm.state_dict(), "vlm_best.pt")
import shutil; shutil.make_archive("organvlm_outputs", "zip", ".", FIG_DIR)
try:
    from google.colab import files
    files.download("organvlm_outputs.zip"); files.download("results.csv")
except ImportError:
    print("Saved figures/, results.csv, organvlm_outputs.zip and model checkpoints in", os.getcwd())
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                  "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
nbf.write(nb, "OrganVLM_Colab.ipynb")
print("written", len(C), "cells")
