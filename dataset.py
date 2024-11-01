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
from torchvision import transforms


## create a simple dataset class

class fMRI_Augmentation_Dataset(Dataset):

    """Here images are not included to avoid memory issues, since they're not used in the model"""
    def __init__(self, fmri_data,features,subject_id):
        self.data = fmri_data
        self.features = features
        self.subject_id = subject_id


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data= self.data[idx]

        features = self.features[idx]
        subject_id = self.subject_id

        output = {'data': data,  'features': features, 'subject_id': subject_id}
        return output
        # return data,features,images,subject_id




class fMRI_Dataset(Dataset):
    def __init__(self, fmri_data,images,features,subject_id):
        self.data = fmri_data
        self.images = images
        self.features = features
        self.subject_id = subject_id
        self.image_transform = transforms.ToTensor()  # Define the transform to convert PIL images to tensors


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data= self.data[idx]
        images = self.image_transform(self.images[idx])

        features = self.features[idx]
        subject_id = self.subject_id

        output = {'data': data, 'images': images, 'features': features, 'subject_id': subject_id}
        return output
        # return data,features,images,subject_id


class fMRI_Text_Dataset(Dataset):
    def __init__(self, fmri_data,captions,features,subject_id):
        self.data = fmri_data
        self.captions = captions
        self.features = features
        self.subject_id = subject_id


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data= self.data[idx]
        caption = self.captions[idx]

        features = self.features[idx]
        subject_id = self.subject_id

        output = {'data': data, 'captions': caption, 'features': features, 'subject_id': subject_id}
        return output
        # return data,features,images,subject_id


class fMRI_Multi_Dataset(Dataset):
    def __init__(self, fmri_data,images,captions,image_features,text_features,subject_id,feature_type="image"):
        self.data = fmri_data
        self.images = images
        self.caption = captions
        self.image_features = image_features
        self.text_features = text_features
        self.subject_id = subject_id
        self.image_transform = transforms.ToTensor()  # Define the transform to convert PIL images to tensors
        self.feature_type = feature_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data= self.data[idx]
        images = self.image_transform(self.images[idx])
        caption = self.caption[idx]

        image_features = self.image_features[idx]
        text_features = self.text_features[idx]

        if self.feature_type == "image":
            features = image_features
        elif self.feature_type == "text":
            features = text_features
        subject_id = self.subject_id

        output = {'data': data, 'images': images, 'captions': caption, 'features': features, 'text_features':text_features, 'image_features': image_features, 'subject_id': subject_id}
        return output
        # return data,features,images,subject_id
