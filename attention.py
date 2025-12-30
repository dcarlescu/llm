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