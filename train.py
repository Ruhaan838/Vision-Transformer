import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torch import functional as F

from sklearn import metrics
import pandas as pd

from model.get_model import GetModel
from config import Config,GPUConfig
from Dataset.dataset import get_dataloader


def train(model: nn.Module, dataloader:DataLoader, optim:optim, loss_fn:nn, device:torch.device) -> list:
    model.train()
    pbar = tqdm(dataloader, desc="Training")
    
    for image,label in pbar:
        
        image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)
        
        pred = model(image)
        
        loss = loss_fn(pred, label)
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        pred = pred.argmax(dim=1)
        
        label, pred = label.to("cpu"), pred.to("cpu")
        acc = metrics.accuracy_score(label.numpy(),pred.numpy())
        precision = metrics.precision_score(label.numpy(), pred.numpy(), average='weighted', zero_division=0)
        recall = metrics.recall_score(label.numpy(), pred.numpy(), average='weighted', zero_division=0)
        
        pbar.set_postfix(Accuracy = acc, Loss = loss.item(), Precision = precision, Recall = recall)
    
    return acc, loss.item(), precision, recall

def valid(model:nn.Module,val_loader:DataLoader, device:torch.device):
    model.eval()
    val_loss = 0 
    with torch.no_grad():
        for image,labels in val_loader:
            
            image , labels = image.to(device, non_blocking=True),labels.to(device, non_blocking=True)
            output = model(image)
            
            val_loss += F.cross_entropy(output, labels, reduction='sum').item() 
            pred = output.argmax(dim=1,keepdim=True)
            
            labels , pred = labels.to('cpu') , pred.to('cpu')
            
            acc = metrics.accuracy_score(labels.numpy(),pred.numpy())
            precision = metrics.precision_score(labels.numpy(),pred.numpy() , average='weighted', zero_division=0)
            recall = metrics.recall_score(labels.numpy(),pred.numpy(), average='weighted', zero_division=0)
            
    val_loss /= len(val_loader.dataset)
    
    
    print(f'|| Validation Loss = {val_loss:0.4f} || Validation Accuracy = {acc:0.4f} || Validation Precision = {precision:0.4f} || Validation Recall = {recall:0.4f} ||')
    
    return acc, val_loss, precision, recall

def update_metrics(metrics:dict, train_list:list, val_list) -> None:
    metrics['accuracy'].append(train_list[0])
    metrics['val_accuracy'].append(val_list[0])
    
    metrics['loss'].append(train_list[1])
    metrics['val_loss'].append(val_list[1])
    
    metrics['precision'].append(train_list[2])
    metrics['val_precision'].append(val_list[2])
    
    metrics['recall'].append(train_list[3])
    metrics['val_recall'].append(val_list[3])

if __name__ == "__main__":
    model = GetModel.get_model()
    optimizer = optim.AdamW(model.parameters(),lr=Config.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss().to(GPUConfig.device)
    metric = {
    'accuracy':[],'val_accuracy':[],
    'loss':[],'val_loss':[],
    'precision':[],'val_precision':[],
    'recall':[],'val_recall':[]
    }
    
    train_dataloader, val_dataloader = get_dataloader()
    
    for epoch in range(Config.EPOCH):
        print(f'EPOCH: [{epoch+1} / {Config.EPOCH}]')
        train_ans = train(model, train_dataloader, optimizer, loss_fn, GPUConfig.device)
        val_ans = valid(model, val_dataloader, GPUConfig.device)
        update_metrics(metric, train_ans, val_ans) # temp function for look train and valid loop clear

    df = pd.DataFrame(metric)
    df.to_csv("results")