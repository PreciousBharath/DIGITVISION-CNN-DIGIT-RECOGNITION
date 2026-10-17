"""DigitVision engine: PyTorch CNN on handwritten digits, augmentation, training loop, baselines, preprocessing."""
import base64
import io
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

torch.set_num_threads(max(1, torch.get_num_threads()))
SIZE = 16  # digits are upsampled from 8x8 to 16x16 for a deeper CNN


def load_data(seed=0):
    d = load_digits()
    X = d.images.astype(np.float32) / 16.0
    X = np.stack([np.array(Image.fromarray((x * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)) / 255.0 for x in X]).astype(np.float32)
    return train_test_split(X, d.target, test_size=0.2, stratify=d.target, random_state=seed)


class DigitCNN(nn.Module):
    def __init__(self, dropout=0.3, width=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, width, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(width * 2)
        self.conv3 = nn.Conv2d(width * 2, width * 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(width * 4 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        a1 = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)   # 8x8
        a2 = F.max_pool2d(F.relu(self.bn2(self.conv2(a1))), 2)  # 4x4
        a3 = F.max_pool2d(F.relu(self.bn3(self.conv3(a2))), 2)  # 2x2
        return a1, a2, a3

    def embed(self, x):
        return F.relu(self.fc1(torch.flatten(self.features(x)[2], 1)))

    def forward(self, x):
        return self.fc2(self.drop(self.embed(x)))


def augment(xb, strength=1.0):
    """Random affine jitter (rotation, scale, shift) + noise, applied on the GPU/CPU tensor batch."""
    n = xb.shape[0]
    ang = (torch.rand(n) - 0.5) * 0.5 * strength
    sc = 1 + (torch.rand(n) - 0.5) * 0.25 * strength
    tx, ty = [(torch.rand(n) - 0.5) * 0.3 * strength for _ in range(2)]
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = torch.cos(ang) / sc, -torch.sin(ang) / sc, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = torch.sin(ang) / sc, torch.cos(ang) / sc, ty
    grid = F.affine_grid(theta, xb.shape, align_corners=False)
    out = F.grid_sample(xb, grid, align_corners=False, padding_mode="zeros")
    return (out + torch.randn_like(out) * 0.05 * strength).clamp(0, 1)


def train_cnn(Xtr, ytr, Xte, yte, epochs=25, lr=2e-3, batch=64, dropout=0.3, aug=1.0, width=32, seed=0, callback=None):
    torch.manual_seed(seed)
    model = DigitCNN(dropout, width)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(Xtr) / batch)))
    xt, yt = torch.tensor(Xtr).unsqueeze(1), torch.tensor(ytr)
    xv, yv = torch.tensor(Xte).unsqueeze(1), torch.tensor(yte)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        tot, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            xb = augment(xt[idx], aug) if aug > 0 else xt[idx]
            out = model(xb)
            loss = F.cross_entropy(out, yt[idx], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            loss_sum += loss.item() * len(idx); tot += len(idx); correct += (out.argmax(1) == yt[idx]).sum().item()
        model.eval()
        with torch.no_grad():
            vo = model(xv)
            vloss = F.cross_entropy(vo, yv).item()
            vacc = (vo.argmax(1) == yv).float().mean().item()
        hist.append({"epoch": ep + 1, "train_loss": loss_sum / tot, "val_loss": vloss, "train_acc": correct / tot, "val_acc": vacc})
        if callback:
            callback(hist)
    model.eval()
    return model, hist


def predict(model, imgs):
    with torch.no_grad():
        x = torch.tensor(np.asarray(imgs, dtype=np.float32)).reshape(-1, 1, SIZE, SIZE)
        return F.softmax(model(x), 1).numpy()


def baselines(Xtr, ytr, Xte, yte):
    out = {}
    for name, m in {"Logistic Regression": LogisticRegression(max_iter=2000), "k-NN (k=3)": KNeighborsClassifier(3), "SVM (RBF)": SVC(gamma="scale", C=5)}.items():
        t = time.time(); m.fit(Xtr.reshape(len(Xtr), -1), ytr)
        out[name] = {"acc": m.score(Xte.reshape(len(Xte), -1), yte), "time": time.time() - t}
    return out


def confusion(model, Xte, yte):
    pred = predict(model, Xte).argmax(1)
    return confusion_matrix(yte, pred), pred


def embedding_2d(model, X):
    with torch.no_grad():
        e = model.embed(torch.tensor(X).unsqueeze(1)).numpy()
    return PCA(2, random_state=0).fit_transform(e)


def feature_maps(model, img):
    with torch.no_grad():
        a1, a2, _ = model.features(torch.tensor(img, dtype=torch.float32).reshape(1, 1, SIZE, SIZE))
    return a1[0].numpy(), a2[0].numpy()


def preprocess(pil: Image.Image, invert="auto"):
    """Photo / drawing → centred 16x16 digit like the training data. Returns (final, steps dict)."""
    steps = {}
    g = ImageOps.grayscale(pil.convert("RGBA").convert("RGB")) if pil.mode != "L" else pil
    steps["grayscale"] = g
    arr = np.asarray(g, dtype=np.float32)
    if invert == "auto":
        invert = arr.mean() > 127  # dark ink on light paper → invert
    if invert:
        arr = 255 - arr
    arr[arr < 40] = 0
    steps["ink"] = Image.fromarray(arr.astype(np.uint8))
    ys, xs = np.nonzero(arr > 50)
    if len(xs) == 0:
        return None, steps
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = int(max(h, w) * 1.3)
    sq = np.zeros((side, side), np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    steps["cropped"] = Image.fromarray(sq.astype(np.uint8))
    small = Image.fromarray(sq.astype(np.uint8)).resize((8, 8), Image.LANCZOS).resize((SIZE, SIZE), Image.BILINEAR)
    out = np.asarray(small, dtype=np.float32)
    out = out / (out.max() + 1e-6)
    steps["model input"] = Image.fromarray((out * 255).astype(np.uint8)).resize((128, 128), Image.NEAREST)
    return out, steps


def decode_data_url(url):
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
