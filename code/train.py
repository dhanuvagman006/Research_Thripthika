"""Organ classification on OrganMNIST3D: baseline 3D CNN vs. a simple vision-language model.

Usage:
    python train.py --model baseline        # standard 3D CNN with cross-entropy
    python train.py --model vlm             # 3D CNN + ClinicalBERT text matching with SigLIP loss
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medmnist import OrganMNIST3D
from sklearn.metrics import confusion_matrix, roc_auc_score

ORGANS = ["liver", "right kidney", "left kidney", "right femur", "left femur", "bladder",
          "heart", "right lung", "left lung", "spleen", "pancreas"]


def load_split(split, root):
    ds = OrganMNIST3D(split=split, download=True, root=root)
    x = torch.tensor(ds.imgs, dtype=torch.float32).unsqueeze(1)  # (N, 1, 28, 28, 28)
    if x.max() > 1:
        x = x / 255.0                                            # scale voxels to 0-1
    y = torch.tensor(ds.labels).long().squeeze(1)
    return x, y


def cnn_encoder():
    """Four blocks of 3x3x3 convolution, batch norm, ReLU and max pooling -> 256 features."""
    channels, layers = [1, 32, 64, 128, 256], []
    for i in range(4):
        layers += [nn.Conv3d(channels[i], channels[i + 1], 3, padding=1),
                   nn.BatchNorm3d(channels[i + 1]), nn.ReLU()]
        if i < 3:
            layers.append(nn.MaxPool3d(2))
    return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten())


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = cnn_encoder()
        self.fc = nn.Linear(256, len(ORGANS))

    def forward(self, x):
        return self.fc(self.encoder(x))


class OrganVLM(nn.Module):
    """Scores each scan against the 11 organ sentences; the highest score is the prediction."""

    def __init__(self, text_vectors):
        super().__init__()
        self.encoder = cnn_encoder()
        self.proj = nn.Linear(256, text_vectors.shape[1])
        self.register_buffer("text", F.normalize(text_vectors, dim=-1))
        self.log_t = nn.Parameter(torch.tensor(np.log(10.0)))   # learnable scale t
        self.b = nn.Parameter(torch.tensor(-10.0))               # learnable bias b

    def forward(self, x):
        image = F.normalize(self.proj(self.encoder(x)), dim=-1)
        return self.log_t.exp() * image @ self.text.T + self.b  # t * cosine similarity + b


def organ_text_vectors(device):
    """Frozen ClinicalBERT [CLS] vectors for 'A CT scan of the <organ>.' (768 numbers each)."""
    from transformers import AutoModel, AutoTokenizer
    name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer, bert = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
    with torch.no_grad():
        batch = tokenizer([f"A CT scan of the {o}." for o in ORGANS], padding=True, return_tensors="pt")
        vectors = bert(**batch).last_hidden_state[:, 0]
    # The 11 sentences differ by one word, so their raw vectors are almost identical
    # (cosine ~0.98). Subtracting their mean (centring) makes them distinct.
    return (vectors - vectors.mean(0, keepdim=True)).to(device)


def siglip_loss(logits, y):
    z = -torch.ones_like(logits)
    z[torch.arange(len(y)), y] = 1                               # +1 for the correct organ, -1 otherwise
    return -F.logsigmoid(z * logits).sum(1).mean()


def random_flip(x):
    for dim in (2, 3, 4):
        flip = torch.rand(len(x), device=x.device) < 0.5
        x[flip] = x[flip].flip(dim)
    return x


@torch.no_grad()
def evaluate(model, x, y, device):
    model.eval()
    logits = torch.cat([model(x[i:i + 128].to(device)).cpu() for i in range(0, len(x), 128)])
    pred = logits.argmax(1)
    acc = (pred == y).float().mean().item()
    auc = roc_auc_score(y.numpy(), logits.softmax(1).numpy(), multi_class="ovr")
    return acc, auc, pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "vlm"], default="baseline")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import os
    os.makedirs(args.data_dir, exist_ok=True)
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = (load_split(s, args.data_dir) for s in ("train", "val", "test"))
    print(f"train {len(x_tr)} | val {len(x_va)} | test {len(x_te)} | device {device}")

    if args.model == "baseline":
        model, loss_fn = BaselineCNN(), F.cross_entropy
    else:
        model, loss_fn = OrganVLM(organ_text_vectors(device)), siglip_loss
    model = model.to(device)
    print(f"model {args.model} | trainable parameters {sum(q.numel() for q in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc, best_state, start = -1.0, None, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), args.batch_size):
            idx = order[i:i + args.batch_size]
            xb, yb = random_flip(x_tr[idx].to(device)), y_tr[idx].to(device)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc, _, _ = evaluate(model, x_va, y_va, device)
        if val_acc > best_acc:                                   # keep the best model on validation
            best_acc, best_state = val_acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d} | loss {loss.item():.4f} | val acc {val_acc:.3f}")

    model.load_state_dict(best_state)
    torch.save(best_state, f"{args.model}_best.pt")
    test_acc, test_auc, pred = evaluate(model, x_te, y_te, device)
    print(f"\nTEST accuracy {test_acc:.3f} | AUC {test_auc:.3f} | best val acc {best_acc:.3f} | "
          f"time {(time.time() - start) / 60:.1f} min")
    print("confusion matrix (rows = true organ, columns = predicted):")
    print(confusion_matrix(y_te.numpy(), pred.numpy()))


if __name__ == "__main__":
    main()
