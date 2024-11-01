import numpy as np
import nibabel as nib
import nilearn 
import matplotlib.pyplot as plt
import os
from os.path import join as opj
import pandas as pd
import seaborn as sns
import glob
from nilearn import plotting
from nilearn.image import *
import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from nilearn.plotting import plot_stat_map
from nilearn.image import mean_img
from nilearn.plotting import plot_img, plot_epi
from nilearn.maskers import NiftiMasker
from sklearn.preprocessing import StandardScaler
import wandb
import pickle
from torch.utils.data import Dataset, DataLoader
from dataset import fMRI_Dataset
import torch
from torch import nn
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger

import torch
import torch.nn as nn
import pytorch_lightning as pl

class ContrastiveModel(pl.LightningModule):

    def contrastive_loss(self, z_i, z_j, normalize =True):
        if normalize:
            z_i = nn.functional.normalize(z_i, dim=1)
            z_j = nn.functional.normalize(z_j, dim=1)
        
        logits = (z_i @ z_j.T) / self.temperature
        similarities = z_j @ z_j.T
        # targets = torch.nn.functional.softmax(similarities * self.temperature, dim=-1)

        targets = torch.arange(logits.shape[0]).long().to(logits.device)
        
        loss = 0.5*torch.nn.functional.cross_entropy(logits, targets) + 0.5*torch.nn.functional.cross_entropy(logits.T, targets)
        return loss
    
    def mean_contrastive(self, z_i, z_j, temperature=1.0):
        return (1-self.alpha)*nn.functional.mse_loss(z_i, z_j)+(self.alpha)*self.contrastive_loss(z_i, z_j)
    
    def cosine_loss(self, z_i, z_j, temperature=1.0):
        cosine_similarity = torch.nn.functional.cosine_similarity(z_i, z_j).mean()
        return 1- cosine_similarity

    def __init__(self,  num_input_channels: int, hidden_dims, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU, temperature=.1,loss_type="contrastive",alpha=0.8, lr=1e-4,wd = 1e-4):
        super().__init__()
        
        self.temperature=temperature
        
        self.model = Encoder(num_input_channels, base_channel_size, latent_dim, act_fn)
        self.alpha = alpha
        self.lr = lr
        self.wd = wd

        self.loss_type=loss_type
        if loss_type=="contrastive":
            self.loss_fn=self.contrastive_loss
        elif loss_type=="mean_contrastive":
            self.loss_fn=self.mean_contrastive
        
        elif loss_type=="mse":
            self.loss_fn=torch.nn.functional.mse_loss
        elif loss_type=="cosine":
            self.loss_fn=self.cosine_loss
        
        self.train_losses = []
        self.train_mse=[]
        self.train_cosine=[]
        self.val_losses = []
        self.val_mse=[]
        self.val_cosine=[]
    
        self.train_history={}
        self.val_history={}

        #init history
        self.train_history["train_loss"]=[]
        self.train_history["train_mse"]=[]
        self.train_history["train_cosine"]=[]
        self.val_history["val_loss"]=[]
        self.val_history["val_mse"]=[]
        self.val_history["val_cosine"]=[]


    def forward(self, x, embeddings = None,**kwargs):
        fmri_proj = self.model(x, **kwargs)
        
        if embeddings is not None:
            embeddings_proj = embeddings   ##Possibly add a layer to adapt this?
            return fmri_proj, embeddings_proj
        
        return fmri_proj


        # return self.model(x,**kwargs)
    
    def training_step(self, batch, batch_idx):
        x = batch["data"]
        embeddings = batch["features"]
        idx = batch["subject_id"]

        # x = x.float()
        y_hat_fmri, y_hat_embeddings = self(x, embeddings, k=idx)
        loss = self.loss_fn(y_hat_fmri, y_hat_embeddings)


        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.item())

        mse_loss = torch.nn.functional.mse_loss(y_hat_fmri, y_hat_embeddings)
        cosine_similarity = torch.nn.functional.cosine_similarity(y_hat_fmri, y_hat_embeddings).mean()
        self.train_mse.append(mse_loss.item())
        self.train_cosine.append(cosine_similarity.item())

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        
        x = batch["data"]
        embeddings = batch["features"]
        idx = batch["subject_id"]

        y_hat_fmri, y_hat_embeddings = self(x, embeddings, k=idx)

        loss = self.loss_fn(y_hat_fmri, y_hat_embeddings)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        mse_loss = torch.nn.functional.mse_loss(y_hat_fmri, y_hat_embeddings)
        self.log('val_mse_loss', mse_loss, on_epoch=True, prog_bar=True)
        
        cosine_similarity = torch.nn.functional.cosine_similarity(y_hat_fmri, y_hat_embeddings).mean()
        self.log('val_cosine_similarity', cosine_similarity, on_epoch=True, prog_bar=True)
        
        self.val_losses.append(loss.item())
        self.val_mse.append(mse_loss.item())
        self.val_cosine.append(cosine_similarity.item())
        return mse_loss
        
    
    def on_train_epoch_end(self) -> None:
        self.train_history["train_loss"].append(np.mean(self.train_losses))
        self.train_history["train_mse"].append(np.mean(self.train_mse))
        self.train_history["train_cosine"].append(np.mean(self.train_cosine))
        self.train_losses = []
        self.train_mse=[]
        self.train_cosine=[]
        return super().on_train_epoch_end()
    
    def on_validation_epoch_end(self) -> None:
        self.val_history["val_loss"].append(np.mean(self.val_losses))
        self.val_history["val_mse"].append(np.mean(self.val_mse))
        self.val_history["val_cosine"].append(np.mean(self.val_cosine))
        self.val_losses = []

        self.val_mse=[]
        self.val_cosine=[]
        return super().on_validation_epoch_end()

        
    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=0)
        # add a scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # use a scheduler that every 100 steps, it will reduce the learning rate by 0.1
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, verbose=True)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,act_fn=nn.ReLU,alignment_layers_keys=[1,2,3,4],common_dim=4096):
        super(Encoder, self).__init__()
        self.common_dim=common_dim
        self.alignment_layers={}
        for k in alignment_layers_keys:
            self.alignment_layers[k]=nn.LazyLinear(common_dim)
        
        
        layers = []
        prev_dim = input_dim
        if len(hidden_dims):
            
            layers.append(nn.LayerNorm(common_dim))
            for hidden_dim in hidden_dims:
                layers.append(nn.LazyLinear(hidden_dim))
                layers.append(act_fn())
                prev_dim = hidden_dim
            
            #another layer norm
            layers.append(nn.LazyLinear(output_dim))
            layers.append(nn.LayerNorm(output_dim))
            
            self.net = nn.Sequential(*layers)     
        else:
            self.net = nn.Linear(input_dim, output_dim)

        
    def _apply(self, fn):
        super(Encoder, self)._apply(fn)        
        for k,v in self.alignment_layers.items():
            self.alignment_layers[k]._apply(fn)
            
    
    def forward(self, x, k=None):
        
        def apply_alignment_layers(x, k, alignment_layers):
            # Create an empty tensor to store the results
            result = torch.empty_like(x)
            result = result[:,:self.common_dim]
            
            # Iterate through each unique key in k
            for key in k.unique():
                # Create a mask for all elements that match the current key
                mask = (k == key.item())
                
                # print(x.shape, result.shape, mask)
                
                # Apply the corresponding alignment layer to the masked elements
                result[mask] = alignment_layers[key.item()](x[mask])
            
            return result
        
        if k is None:
            k=torch.ones(len(x))
        # Apply alignment layers to x using the custom function
        x = apply_alignment_layers(x, k, self.alignment_layers)
        
        # x = self.alignment_layers[k](x)
        return self.net(x)





