import torch
from torch import nn
from .attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float = 0.2) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        x = self.linear_1(x)
        x = self.dropout(x)
        out = self.linear_2(x)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, seq_len:int, head_size:int, dropout:float) -> None:
        super().__init__()

        self.layernorm1 = nn.LayerNorm(d_model)
        self.multiheadattention = MultiHeadAttention(d_model, seq_len, head_size, dropout)
        
        self.layernorm2 = nn.LayerNorm(d_model)
        self.feedforward = FeedForward(d_model ,d_ff, dropout)
        self.active = nn.GELU()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:

        res = x
        x = self.layernorm1(x)
        x = self.multiheadattention(x, x, x)
        x += res

        res = x
        x = self.layernorm2(x)
        x = self.feedforward(x)
        x = self.active(x)
        x += res

        return x
    
