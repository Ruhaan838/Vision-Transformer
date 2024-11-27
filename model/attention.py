import torch
from torch import nn
import math

class SelfAttention(nn.Module):
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:any = None) -> torch.Tensor:
        d_k = q.size(-1)
        attention = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        
        if mask is not None:
            attention.masked_fill_(mask == 0, -1e9)
        
        attention = attention.softmax(dim= -1)

        attention = attention @ v
        
        return attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, seq_len:int, head_size:int, dropout:float = 0.2) -> torch.Tensor:
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.head_size = head_size

        assert d_model % head_size == 0, "d_model is divisible by the head_size"
        self.d_k = d_model // head_size

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.attention = SelfAttention()

        self.dropout = nn.Dropout(dropout)

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask=None) -> torch.Tensor:

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        #(b, seq_len, d_model) -> (b, seq_len, head, d_k) -> (b, head, seq_len, d_k)
        query = q.view(q.size(0), q.size(1), self.head_size, self.d_k).transpose(1, 2) 
        key = k.view(k.size(0), k.size(1), self.head_size, self.d_k).transpose(1, 2)
        value = v.view(v.size(0), v.size(1), self.head_size, self.d_k).transpose(1, 2)

        attention = self.attention(query, key, value, mask)
        
        out = attention.transpose(1, 2).contiguous().view(attention.size(0), -1, self.d_model)

        out = self.wo(out)

        return out