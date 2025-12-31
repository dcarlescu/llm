import torch.nn as nn
import torch

from attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.multi_head_attn = MultiHeadAttention(
                d_in=cfg["emb_dim"], 
                d_out=cfg["emb_dim"], 
                context_length=cfg["context_length"], 
                dropout=cfg["drop_rate"], 
                num_heads=cfg["num_heads"], 
                qkv_bias=False)
        
        self.ln1 = LayerNorm(cfg["emb_dim"])
        self.ln2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        x = self.multi_head_attn(x) #already includes a dropout layer
        x = self.dropout(x)
        x = x + shortcut #add shortcut connection

        shortcut = x #save the input for the next shortcut
        x = self.ln2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut #add shortcut connection
        return x

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # small constant to avoid division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) #use unbiased to not use Bessel's correction
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
            ))    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                GELU(),
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            )
    
    def forward(self, x):
        return self.layers(x)    