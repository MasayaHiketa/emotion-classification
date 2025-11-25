import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained("outputs")
model = AutoModelForSequenceClassification.from_pretrained("outputs").to(device)
model.eval()

id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# 推論関數
def predict_emotion(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = probs.argmax()

    label_name = id2label[pred_id]


    return label_name, probs.tolist()
