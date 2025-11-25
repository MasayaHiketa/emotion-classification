import torch
from torch.optim import AdamW
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from dataset import load_and_tokenize_dataset

from tokenizer import load_tokenizer
from dataset import create_dataloader
from model import load_model
from utils import load_config, set_seed, compute_metrics

def train():
    config = load_config()
    set_seed(42)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # tokenizer
    tokenizer = load_tokenizer(config["model_name"], max_len=config["max_len"])

    #  tokenize 済み dataset をロード
    hf_dataset = load_and_tokenize_dataset(config["train_data"], tokenizer, max_len=config["max_len"])
    hf_dataset = hf_dataset.shuffle(seed=42).select(range(60000))

    # dataloader 作成
    train_loader = create_dataloader(
        hf_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        pin_memory=True
    )



    # model
    model = load_model(config["model_name"], num_labels=6)
    model.to(device)

    #  VRAM節約
    model.gradient_checkpointing_enable()

    lr = float(config["lr"])
    optimizer = AdamW(model.parameters(), lr=lr)

    #  AMP 初期化
    scaler = GradScaler()

    # training loop
    model.train()
    for epoch in range(1, config["epochs"] + 1):
        print(f"\n===== Epoch {epoch}/{config['epochs']} =====")
        all_preds = []
        all_labels = []
        total_loss = 0

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # 🚀 AMP（半精度）で forward
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                logits = outputs.logits

            # 🚀 AMP（半精度）で backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            all_preds.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        # 評価
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        metrics = compute_metrics(all_preds, all_labels)

        print(f"Loss: {total_loss/len(train_loader):.4f}  |  Acc: {metrics['accuracy']:.4f}  |  F1: {metrics['f1']:.4f}")

    # 保存
    output_dir = config["output_dir"]
    model.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    train()
