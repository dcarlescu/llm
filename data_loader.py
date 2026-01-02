import torch
from torch.utils.data import DataLoader, Dataset

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.output_ids = []

        token_ids  = tokenizer.encode(txt)
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            output_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.output_ids.append(torch.tensor(output_chunk))
    
    def __len__(self):
        return len(self.input_ids)
   
    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]
    
def create_dataloader(txt, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

def load_verdict_data(tokenizer):
    file_path = "training-data/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Verdict characters:", total_characters)
    print("Verdict tokens:", total_tokens)

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    return train_data, val_data