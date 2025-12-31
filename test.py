import tiktoken
import torch

from config import GPT_CONFIG_SMALL
from data_loader import create_dataloader
from gpt import GPTModel
from training import train_model


file_path = "training-data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

tokenizer = tiktoken.get_encoding("gpt2")
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)
train_loader = create_dataloader(
                train_data,
                tokenizer,
                batch_size=2,
                max_length=GPT_CONFIG_SMALL["context_length"],
                stride=GPT_CONFIG_SMALL["context_length"],
                drop_last=True,
                shuffle=True,
                num_workers=0)

val_loader = create_dataloader(
                val_data,
                tokenizer,
                batch_size=2,
                max_length=GPT_CONFIG_SMALL["context_length"],
                stride=GPT_CONFIG_SMALL["context_length"],
                drop_last=False,
                shuffle=False,
                num_workers=0)

model = GPTModel(GPT_CONFIG_SMALL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer
)

