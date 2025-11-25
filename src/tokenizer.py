from transformers import AutoTokenizer

def load_tokenizer(model_name: str, max_len: int = 128):
    """
    中文
    載入 HuggingFace Tokenizer（BERT, RoBERTa...）

    日本語補助
    Tokenizer の設定をまとめる。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = max_len
    return tokenizer
