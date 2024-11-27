import torch
from torch import nn
from .embedding import PatchEmbedding
from ..config import Config
from .Encoder import EncoderBlock


class ViT(nn.Module):
    def __init__(self, in_channel: int, num_patch: int, N_block: int, d_model: int, d_ff: int, head_size: int, dropout: float) -> None:
        super().__init__()
        
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model),requires_grad=True)
        
        self.patchembedding = PatchEmbedding(in_channel, num_patch, d_model)
        seq_len = (Config.image_size // Config.patch_size) ** 2
        self.encoders = nn.ModuleList([EncoderBlock(d_model, d_ff, seq_len, head_size, dropout) for _ in range(N_block)])
        
        self.class_layer = nn.Linear(d_model, Config.n_class)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
 
        x = self.patchembedding(x)  
        for layer in self.encoders:
            x = layer(x)
        
        x = x[:, 0, :]
        x = self.class_layer(x)
        
        return x