from torchvision.datasets import ImageFolder
from ..config import Config
from .transform import transform
from torch.utils.data import DataLoader

trans = transform()
def get_dataloader():
    train_dataset = ImageFolder(root = Config.train_dir, transform = trans)
    val_dataset = ImageFolder(root = Config.val_dir, transform = trans)

    train_dataloader = DataLoader(train_dataset, Config.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, Config.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    
    return train_dataloader, val_dataloader