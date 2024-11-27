from torchvision import transforms as T
from ..config import Config

def transform():
    return T.Compose([
        T.Resize((Config.image_size,Config.image_size)),
    #     T.RandomAutocontrast(0.2),
    #     T.CenterCrop((Config.image_size,Config.image_size)),
        T.ToTensor()
    ])