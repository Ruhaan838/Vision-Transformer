# 👁️ Vision Transformer 

This repository contains an implementation of the Vision Transformer (ViT) architecture built from scratch.

## 📄 Reference Paper  

The architecture is based on the paper:  

> **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**  
> *Authors: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, et al.*  
> Published by Google Research, Brain Team.  
> **[Paper link](https://arxiv.org/abs/2010.11929)**

## 🛠️ Features of This Implementation  

- **Tokenization**: Patch-based image splitting to create input tokens.  
- **Transformer Encoder**: Multi-head self-attention and feed-forward layers.  
- **Positional Encoding**: Integration of positional information into input tokens.  
- **Classifier Head**: Fully connected layer for classification tasks.  
- **Modular Design**: Clean and reusable code for core components like tokenizers, encoders, and heads.  

## 🚀 Getting Started  

### Prerequisites  
Ensure you have Python 3.8+ installed and install the `requirements.txt` using 
```bash
pip install -r requirements.txt
``` 

### Running the Code  
1. Clone this repository:  
   ```bash
   git clone https://github.com/Ruhaan838/Vision-Transformer
   cd Vision-Transformer
   ```  

2. Train the ViT on a dataset (e.g., CIFAR-10):  
   ```bash
   python train.py 
   ```  

3. Evaluate the model:  
   ```bash
   python eval.py
   ```  

## 📁 Project Structure  

```plaintext
vision-transformer/  
├── dataset/              # Dataset Classes  
├── models/               # Vision Transformer model components  
│   ├── vit.py            # Main ViT implementation  
│   ├── attention.py      # Scripts for SelfAttention and MultiHeadAttention
│   ├── Encoder.py        # Scripts for Encoder for Vit.
│   ├── get_model.py      # retrun the full model with Config set on config.py
│   ├── embedding.py      # Scripts for patch Embedding 
├── notebooks/
    ├── vit-scratch.ipynb # Jupyter notebook for model training and more ... 
├── config.py             # Configurations of full model
├── train.py              # Training script  
├── eval.py               # Evaluation script  
├── requirements.txt      # Required libraries  
└── README.md             # Project documentation  
```  
Let me know if you’d like assistance with any part of the implementation!