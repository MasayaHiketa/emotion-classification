import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

def tokenize_function(examples, tokenizer, max_len):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_len,
    )

def load_and_tokenize_dataset(parquet_path, tokenizer, max_len=96):
    # parquet を読む
    raw = load_dataset("parquet", data_files={"train": parquet_path})["train"]

    # 🚀 batched=True で高速 tokenize（超速い）
    tokenized = raw.map(
        lambda x: tokenize_function(x, tokenizer, max_len),
        batched=True,
        batch_size=2048,
        remove_columns=["text"]
    )

    return tokenized

class EmotionDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        return {
            "input_ids": torch.tensor(row["input_ids"]),
            "attention_mask": torch.tensor(row["attention_mask"]),
            "labels": torch.tensor(int(row["label"]), dtype=torch.long),
        }

def create_dataloader(hf_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True):
    ds = EmotionDataset(hf_dataset)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

# import torch
# from torch.utils.data import Dataset, DataLoader
# from datasets import load_dataset

# class EmotionDataset(Dataset):
#     """
#     Emotion dataset for parquet → tokenized → PyTorch Dataset
#     """

#     def __init__(self, parquet_path, tokenizer, max_len=128):
#         self.data = load_dataset("parquet", data_files={"train": parquet_path})["train"]
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data[idx]
#         text = row["text"]
#         label = int(row["label"])

#         encoded = self.tokenizer(
#             text,
#             truncation=True,
#             padding="max_length",
#             max_length=self.max_len,
#             return_tensors="pt",
#         )

#         return {
#             "input_ids": encoded["input_ids"].squeeze(0),
#             "attention_mask": encoded["attention_mask"].squeeze(0),
#             "labels": torch.tensor(label, dtype=torch.long),
#         }


# def create_dataloader(parquet_path, tokenizer, batch_size=32, max_len=128, shuffle=True):
#     dataset = EmotionDataset(parquet_path, tokenizer, max_len)
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=2,
#         pin_memory=True,
#     )
