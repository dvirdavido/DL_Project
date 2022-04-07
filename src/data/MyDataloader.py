import os
import sys
import h5py
import hydra
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io.wavfile import write
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.metrics import Precision, Recall, F1, Accuracy
import soundfile
warnings.filterwarnings("ignore", category=UserWarning)
import torchaudio
# torchaudio.set_audio_backend("sox_io")
from data.data_utils import AudioPreProcessing

class AEC_Dataset(Dataset):
    """Generated Input and Output to the model"""

    def __init__(self,args,data_dir,batch_size,starting_point, amount):
        """ 
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.counter = 1
        self.args = args
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.os_listdir = os.listdir(self.data_dir+'\\preprocessing3')
        self.audio_pre_process = AudioPreProcessing(self.args)
        self.starting_point = starting_point
        self.amount = amount

    
    def __len__(self):
        return self.amount
        #### return len(self.os_listdir)
        #return 16

    def __getitem__(self,idx):
        idx_ = idx + self.starting_point
        target_wav  = self.data_dir+'/lables/file_'+str(idx_)+'.csv'
        feature_wav = self.data_dir+'/preprocessing3/file_'+str(idx_)+'.csv'

        return  self.audio_pre_process.transformations(target_wav,feature_wav)

class StftDataModule(pl.LightningDataModule):

    def __init__(self,args): 
        super().__init__()
        self.args=args

    def setup(self,stage=None):
        self.train_set = AEC_Dataset(self.args,self.args.data_set_path,self.args.train_batch_size, self.args.starting_point.train, self.args.amount.train)
        self.val_set = AEC_Dataset(self.args,self.args.data_set_path,self.args.val_batch_size, self.args.starting_point.val, self.args.amount.val) 
        self.test_set = AEC_Dataset(self.args,self.args.data_set_path,self.args.test_batch_size, self.args.starting_point.test, self.args.amount.test)

    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size=self.args.train_batch_size, shuffle=self.args.data_loader_shuffle , num_workers =self.args.num_workers, pin_memory= self.args.pin_memory)
    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.args.val_batch_size, shuffle=self.args.data_loader_shuffle , num_workers = self.args.num_workers, pin_memory= self.args.pin_memory)
    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.args.test_batch_size, shuffle=self.args.data_loader_shuffle , num_workers = self.args.num_workers, pin_memory= self.args.pin_memory)
