import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained("outputs")
model = AutoModelForSequenceClassification.from_pretrained("outputs").to(device)
model.eval()

# label mapping（あなたの config に合わせる）
id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}


def predict(text):
    # tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    # forward
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()

    return {
        "text": text,
        "label": id2label[pred_id],
        "probs": probs.cpu().numpy().tolist()
    }


# sample test
if __name__ == "__main__":
    examples = [
        "I am very happy today",
        "This is terrible",
        "I am scared",
        "I love this so much",
        "I'm extremely angry right now"
    ]

    for t in examples:
        print(predict(t))
