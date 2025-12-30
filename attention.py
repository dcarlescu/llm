import torch.nn as nn
import torch

class SimpleAttention(nn.Module): 
    def __init__(self, dimension_in, dimension_out):
        super().__init__()

        self.W_query = nn.Parameter(torch.rand(dimension_in, dimension_out))
        self.W_key = nn.Parameter(torch.rand(dimension_in, dimension_out))
        self.W_value = nn.Parameter(torch.rand(dimension_in, dimension_out))

    def forward(self, x):
        keys = x @ self.W_key
        values = x @ self.W_value
        queries = x @ self.W_query

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vector = attn_weights @ values
        return context_vector

class SelfAttention(nn.Module): 
    def __init__(self, dimension_in, dimension_out, qvk_bias=False):
        super().__init__()

        self.W_query = nn.Linear(dimension_in, dimension_out, bias=qvk_bias)
        self.W_key = nn.Linear(dimension_in, dimension_out, bias=qvk_bias)
        self.W_value = nn.Linear(dimension_in, dimension_out, bias=qvk_bias)

    def forward(self, x):
        keys = self.W_key(x) #linear transformation does a matrix multiplication inside
        values = self.W_value(x)
        queries = self.W_query(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vector = attn_weights @ values
        return context_vector
    
class CausalAttention(nn.Module): 
    def __init__(self, dimension_in, dimension_out, context_length, dropout, qvk_bias=False):
        super().__init__()

        self.W_query = nn.Linear(dimension_in, dimension_out, bias=qvk_bias)
        self.W_key = nn.Linear(dimension_in, dimension_out, bias=qvk_bias)
        self.W_value = nn.Linear(dimension_in, dimension_out, bias=qvk_bias)
        self.dropout = nn.Dropout(dropout)

        #register the mask to make sure it's on the same device as the model
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) 

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) #linear transformation does a matrix multiplication inside
        values = self.W_value(x)
        queries = self.W_query(x)

        attn_scores = queries @ keys.transpose(1, 2) #transpose just dimension 1 and 2, leave the batch dimension alone

        #apply a mask so that each token can only attend to previous tokens (including itself)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # type: ignore
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) #softmax will turn -inf into 0
        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ values
        return context_vector