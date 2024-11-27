import torch
from torch import nn

from .vit import ViT
from ..config import Config,GPUConfig
from torchinfo import summary

class GetModel:
    
    @staticmethod
    def get_model(is_summary=False):
        model = ViT(
                    in_channel = Config.in_channel, 
                    num_patch = Config.patch_size, 
                    N_block = Config.n_block, 
                    d_model = Config.d_model, 
                    d_ff = Config.d_ff,
                    head_size = Config.head_size,
                    dropout = Config.dropout
                )

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        model = torch.compile(model)
        model = GPUConfig.load_to_gpu(model, GPUConfig.device)
        if is_summary:
            summary(model, (Config.batch_size,Config.in_channel,Config.image_size,Config.image_size))
            
        return model