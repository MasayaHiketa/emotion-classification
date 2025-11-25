from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # ← あなたの model_name
tokenizer.save_pretrained("outputs")
