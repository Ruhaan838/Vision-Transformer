import torch 
import matplotlib.pyplot as plt
import pandas as pd
from config import Config,GPUConfig

def plot_graph(ax, metrics1:list, metrics2:list, label:str) -> None:
    ax.plot(range(1, len(metrics1) + 1), metrics1, label=f'Training {label}', color='green', marker='o')
    ax.plot(range(1, len(metrics2) + 1), metrics2, label=f'Validation {label}', color='red', marker='o')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(f'{label.capitalize()}')
    ax.set_title(f'Training and validation {label.capitalize()}')
    ax.legend()
    ax.grid(True)

def show_prediction(model, dataloader):
    
    images,labels,preds = [],[],[]
    
    with torch.no_grad():
        for image,label in dataloader:
            image,label = image.to(GPUConfig.device),label.to(GPUConfig.device)
            
            pred = model(image).argmax(dim=1)
            
            images.append(image.cpu())
            labels.append(label.cpu().numpy().tolist())
            preds.append(pred.cpu().numpy().tolist())
            break
            
    fig , ax = plt.subplots(5,5,figsize=(25,25))
    ax = ax.flatten()
    
    images,labels,preds = images[0],labels[0],preds[0]
    for i in range(25):
        ax[i].imshow(images[i].permute(1, 2, 0))  
        color = 'green' if labels[i] == preds[i] else 'red'
        ax[i].set_title(
            f'ACTUAL: {Config.train_class[labels[i]]} \nPREDICTION: {Config.train_class[preds[i]]}',
            color=color
        )
        ax[i].axis('off') 
    plt.show()

if __name__ == "__main__":
    
    metric = pd.read_csv('results').to_dict()
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 3))

    # accuracy
    plot_graph(ax1, metric['accuracy'], metric['val_accuracy'], 'accuracy')

    # loss
    plot_graph(ax2, metric['loss'], metric['val_loss'], 'loss')

    # precision
    plot_graph(ax3, metric['precision'], metric['val_precision'], 'precision')

    # recall
    plot_graph(ax4, metric['recall'], metric['val_recall'], 'recall')

    fig.suptitle('Training and validation metrics', fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    plt.show()
    