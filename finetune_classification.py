from pathlib import Path
import time
import pandas as pd
import torch
import tiktoken

from torch.utils.data import DataLoader
from data_loader import SpamDataset, random_split
from openai import OPENAI_CONFIG, load_gpt_model

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
    num_spam, random_state=123
    )
    balanced_df = pd.concat([
    ham_subset, df[df["Label"] == "spam"]
    ])
    return balanced_df

def save_dataset_splits(df):
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    train_df.to_csv("training-data/spam-train.csv", index=None)
    validation_df.to_csv("training-data/spam-validation.csv", index=None)
    test_df.to_csv("training-data/spam-test.csv", index=None)


def freeze_gpt_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def add_classification_head(model, num_classes):
    #replace the output head with a classification head
    model.out_head = torch.nn.Linear(in_features=OPENAI_CONFIG["emb_dim"], out_features=num_classes)

def unfreeze_last_transformer_block(model):
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

tokenizer = tiktoken.get_encoding("gpt2")
#print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

train_dataset = SpamDataset(
    csv_file="training-data/spam-train.csv",
    max_length=None,
    tokenizer=tokenizer
    )
val_dataset = SpamDataset(
    csv_file="training-data/spam-validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
test_dataset = SpamDataset(
    csv_file="training-data/spam-test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
print(train_dataset.max_length)

num_workers = 0
batch_size = 8
torch.manual_seed(123)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

#print(f"{len(train_loader)} training batches")
#print(f"{len(val_loader)} validation batches")
#print(f"{len(test_loader)} test batches")
device = "cpu"
gpt = load_gpt_model(device)
torch.manual_seed(123)

freeze_gpt_parameters(gpt)
add_classification_head(gpt, num_classes=2)
unfreeze_last_transformer_block(gpt)

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
                predicted_labels = torch.argmax(logits, dim=-1)
                num_examples += predicted_labels.shape[0]
                correct_predictions += (
                    (predicted_labels == target_batch).sum().item()
                )
        else:
            break
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
            input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                        num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    return train_losses, val_losses, train_accs, val_accs, examples_seen

def test_evaluation():
    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(
    train_loader, gpt, device, num_batches=10
    )
    val_accuracy = calc_accuracy_loader(
    val_loader, gpt, device, num_batches=10
    )
    test_accuracy = calc_accuracy_loader(
    test_loader, gpt, device, num_batches=10
    )
    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, gpt, device, num_batches=5
        )
    val_loss = calc_loss_loader(val_loader, gpt, device, num_batches=5)
    test_loss = calc_loss_loader(test_loader, gpt, device, num_batches=5)
    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

def run_finetuning():
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        gpt, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50,
        eval_iter=5
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")    

run_finetuning()
test_evaluation()