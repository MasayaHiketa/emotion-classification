import yaml
import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics(preds, labels):
    """
    中文
    計算 accuracy / macro F1。

    日本語補助
    6 クラスなら macro F1 が推奨。
    """
    preds = preds.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}
