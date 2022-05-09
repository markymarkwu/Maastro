## TORCH LIBRARY
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchio as tio

## MONAI LIBRARY
from monai.networks.nets import BasicUNet
from monai.networks.nets import VNet
from monai.networks.nets import UNet
from monai.losses.dice import DiceLoss

## OTHER LIBRARIES
import argparse
import logging
import sys
import numpy as np
from pathlib import Path
import os
import itertools
import matplotlib.pyplot as plt
from dicom_contour.contour import *
from functools import reduce
import SimpleITK as sk
import pydicom as pyd
import copy
import itk
from numpy import savez_compressed
from numpy import load
import multiprocess as mp
import seaborn as sns
import time

## Dataset class
from torch.utils.data import Dataset
import numpy.typing as npt
import pydicom
from rt_utils import RTStructBuilder
from functools import reduce
import random
from numpy import load


## WEIDGHTS & BIASES
import wandb
os.environ["WANDB_CONFIG_DIR"] = "/tmp"
from tqdm import tqdm

class DosePredictionDataset(Dataset):
    def __init__(self, npz_pathlist_patients:list):
        self.patient_folder_paths = npz_pathlist_patients
        self.seed_count = 0

    def __len__(self):
        return len(self.patient_folder_paths)

    def __getitem__(self, index:int):
        patient_dir = self.patient_folder_paths[index] #get patient folder path
        files = []
        for path in os.listdir(patient_dir):
            files.append(os.path.join(patient_dir, path))
        X = torch.from_numpy(load(files[1],allow_pickle=False)['arr_0'])
        y = load(files[0],allow_pickle=False)['arr_0']
        y[0] = y[0]/100
        X = self.rescale_intensity(X)
        X = self.transform(X)
        self.seed_count+=1 
        # print(f"Input data loaded for Patient ID:{patient_dir.split('/')[-1]}")
        return X,y


    ## FUNCTIO TO RESCALE INTENSITY OF PIXEL VALUE
    def rescale_intensity(self, X:torch.TensorType, out_min:float=0.0, out_max:float=1.0):
        in_min = -1024
        in_max = 3072
        tio_image = tio.ScalarImage(tensor=X)
        rescale = tio.RescaleIntensity(in_min_max=(in_min,in_max), 
                                       out_min_max=(out_min, out_max))
        rescaled_image = rescale(tio_image).tensor
        return rescaled_image

    ## FUNCTION TO AUGMENT INPUT
    def transform(self, X:torch.TensorType):
        random.seed(self.seed_count)
        transform = tio.Compose([
            tio.RandomAffine(scales=(random.uniform(0.0,0.5),
                                     random.uniform(0.0,0.5),
                                     random.uniform(0.0,0.5)),
                             degrees=random.randint(0,30),
                             translation=(random.randint(0,10),
                                          random.randint(0,10),
                                          random.randint(0,10))),
            tio.RandomBlur(std=(random.randint(0,10),
                                random.randint(0,10), 
                                random.randint(0,10))),
            tio.RandomNoise(mean=random.uniform(0.0,1.0),
                            std=(0, random.uniform(0,0.5))),
            tio.RandomSwap(patch_size=random.randint(0,5))
        ])
        return transform(X)
       
    ## FUNCTION TO EXTRACT SUBFOLDER PATH
    def get_subfolder_path(self, folder_path):
        subfolder_path = []
        for roots, dirs, files in os.walk(folder_path):
            subfolder_path.append(roots)
        return subfolder_path


## MAIN TRAINING FUNCTION
def train(device:str,
          npz_pathlist_patients:list,
          epochs:list,
          learning_rate:list,
          dropout_rate:list,
          batch_size:list,
          ROI_names:list,
          val_percent: float = 0.4,
          ):
    ## Create dataloaders for training and validation
    dataset = DosePredictionDataset(npz_pathlist_patients)
    val_num = int(len(dataset) * val_percent)
    train_num = len(dataset) - val_num
    train_set, val_set = random_split(dataset, [train_num, val_num], generator=torch.Generator().manual_seed(42)) 
    # Login wandb account
    wandb.login() 
    # Create combinations of hyperparameters
    param_combinations = list(itertools.product(epochs, batch_size, dropout_rate, learning_rate))
    # Create a variable to store the best model and path to store to
    best_val_loss = float('inf')
    best_model = None
    best_param = None
    model_save_path = "Maastro/saved_models/"
    net = None
    ### ================================== TRAINING START ===================================== ###
    print("TRAINING STARTS ...")
    for comb in param_combinations:
        # Initiate and configure wandb runner
        run = wandb.init(reinit=True, project="Maastro")
        #Create neural network model
        net = BasicUNet(spatial_dims=3,
                 in_channels=len(ROI_names)+1, 
                 out_channels=2, 
                 features=(2, 2, 4, 8, 16, 2),
                 dropout=comb[2])
        # net = UNet(spatial_dims=3,
        #            in_channels=len(ROI_names)+1, 
        #            out_channels=2,
        #            channels=(2, 2, 4, 8, 16),
        #            strides=(1,1,1,1))
        run.config.update({"epoch":comb[0],
                           "batch_size":comb[1],
                           "dropout_rate":comb[2],
                           "learning_rate":comb[3],
                           "Input":str(ROI_names),
                           "Model":net.__class__.__name__})
        wandb.watch(net, log='all', log_freq=1)
        train_loader = DataLoader(train_set, shuffle=True, batch_size=run.config.batch_size)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size=run.config.batch_size)
        # Set up optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=run.config.learning_rate) 
        # Create step counter
        sample_count_train = 0
        sample_count_val = 0
        BCE = nn.BCELoss()
        sigmoid = nn.Sigmoid()
        for epoch in tqdm(range(run.config.epoch)):
            # Create epoch loss log variables
            epoch_loss_train = 0
            epoch_loss_val = 0  
            ## ========================== TRAINING SECTION =========================== ##
            net.train()
            for images, masks in train_loader:
                images = images.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.float32)
                # forward pass
                image_pred = net(images)
                # train_loss = DiceLoss().forward(normalize(image_pred[0][0]), masks[0][0])+BCE(sigmoid(image_pred[0][1]), masks[0][1])
                train_loss = get_loss(image_pred, masks)
                epoch_loss_train += train_loss.item()
                # backword pass
                optimizer.zero_grad()
                train_loss.backward()
                # optimizing
                optimizer.step()
                sample_count_train += len(images)
                # Log training loss
                wandb.log({"training_loss": train_loss.item(),
                           "epoch":epoch+1},
                          step=sample_count_train)
            ## ========================== VALIDATION SECTION ========================== ##
            with torch.no_grad():
                sample_count_val = sample_count_train
                for images, masks in val_loader:
                    images = images.to(device=device, dtype=torch.float32)
                    masks = masks.to(device=device, dtype=torch.float32)
                    image_pred = net(images)
                    # val_loss = DiceLoss().forward(normalize(image_pred[0][0]), masks[0][0])+BCE(sigmoid(image_pred[0][1]), masks[0][1])
                    val_loss = get_loss(image_pred, masks)
                    epoch_loss_val += val_loss.item()
                    sample_count_val += len(images)
                    wandb.log({"validation_loss": val_loss.item(),
                               "epoch":epoch+1},
                              step=sample_count_val)

            print(f"Training loss after epoch {epoch+1}: {epoch_loss_train*run.config.batch_size/train_num}"+"||"+ 
                  f"Validation loss after epoch {epoch+1}: {epoch_loss_val*run.config.batch_size/val_num}")                


            if best_val_loss >= epoch_loss_val:
                best_val_loss = epoch_loss_val
                best_model = copy.deepcopy(net)  
                best_param = comb
        run.finish()
    print("TRAINING FINISHED")
    print("Best validation loss is: ", best_val_loss)
    best_param = {'epoch':best_param[0],
                  'batch_size':best_param[1],
                  'dropout_rate':best_param[2],
                  'learning_rate':comb[3]}
    torch.save({'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'param': best_param,
                'loss': best_val_loss}, model_save_path+f"{net.__class__.__name__}_{ROI_names}.pt")


## NORMALIZE TENSOR
def normalize(inputs:torch.TensorType):
    return (inputs-inputs.min())/(inputs.max()-inputs.min())


## GET PATH LIST FOR ALL PATIENTS 
def get_patient_list(top_dir:str):
    pathlist_patients = []
    for folder in os.listdir(top_dir):
        folder_path = os.path.join(top_dir, folder)
        if os.path.isdir(folder_path):
            pathlist_patients.append(folder_path)
    return pathlist_patients


def get_loss(prediction, ground_truth):
    print(prediction.shape[0])
    BCE = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    loss = DiceLoss().forward(normalize(prediction[:,0,:,:,:]), ground_truth[:,0,:,:,:])+BCE(sigmoid(prediction[:,1,:,:,:]), ground_truth[:,1,:,:,:])    
    return loss


## define parameters
ROI_names = ["CTV1"] 
top_dir = "/Users/wangyangwu/Documents/Maastro/NeuralNets/PROTON_FULL"
npz_dir = "Maastro/Data/preprocessed_data_compressed_CTV"
dropout_rate = [0.5]
learning_rate = [0.1]
epochs = [2]
batch_size = [8]


def run():
	npz_pathlist_patients = get_patient_list(npz_dir)
	train("cpu", 
      npz_pathlist_patients, 
      epochs, 
      learning_rate, 
      dropout_rate, 
      batch_size, 
      ROI_names) 

if __name__ is "__main__":
	run()