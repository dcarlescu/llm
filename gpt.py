import torch
import torch.nn as nn

from transformer import LayerNorm, TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  
    
    def print_model_summary(self):
        total_param_count = self.count_parameters()
        total_size_mb = total_param_count * 4 / (1024 ** 2)  # assuming 4 bytes per parameter (float32)
        print(f"Model size: {total_size_mb:.2f} MB")
        print(f"Total number of parameters: {total_param_count:,}")
        feed_forward_params = sum(p.numel() for p in self.trf_blocks[0].ff.parameters() if p.requires_grad) # type: ignore
        print(f"Number of parameters in feed-forward layer of one transformer block: {feed_forward_params:,}")
        attention_params = sum(p.numel() for p in self.trf_blocks[0].multi_head_attn.parameters() if p.requires_grad) # type: ignore
        print(f"Number of parameters in multi-head attention layer of one transformer block: {attention_params:,}") 