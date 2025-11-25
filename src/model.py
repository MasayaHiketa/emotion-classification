import torch.nn as nn
from transformers import AutoModelForSequenceClassification

def load_model(model_name: str, num_labels: int = 6):
    """
    中文
    建立 BERT 系列分類器。

    日本語補助
    AutoModelForSequenceClassification を使う。
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model
