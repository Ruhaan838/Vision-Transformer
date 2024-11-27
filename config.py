import numpy as np
import os
import torch
from torch import nn

class Config:
    train_dir :str = '/kaggle/input/indian-birds/Birds_25/train'
    val_dir :str = '/kaggle/input/indian-birds/Birds_25/valid'
    
    train_class :np.array = np.array(sorted(os.listdir(train_dir)))
    n_class :int = len(train_class)
    
    in_channel :int = 3
    image_size :int = 224
    patch_size :int = 16
    d_model :int = 1024
    d_ff :int = 3072
    batch_size :int = 64
    
    lr :float = 1e-3
    dropout :float = 0.4

    n_block :int = 8
    head_size :int = 16
    
    EPOCH:int = 40
    
class GPUConfig:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    @staticmethod
    def load_to_gpu(model:nn.Module, device:torch.device) -> nn.Module:
        count = torch.cuda.device_count()
        if count > 1:
            print(f"Using {count} {device} device !!")
            model = model.to(device)
            model = nn.DataParallel(model)
            return model
        else:
            print(f"Using {device} device !!")
            model = model.to(device)
            return model