import sys
sys.path.append('/data/ssd/ofersc/NN_AEC_RLS/src/')
sys.path.append('/data/ssd/ofersc/NN_AEC_RLS/references/AEC_2022/')

import numpy as np
import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorch_lightning as pl
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
from scipy.io.wavfile import write
import soundfile
import librosa
from def_model import *
from collections import OrderedDict
import glob
import pysepm
import os
import h5py
import subprocess
from enhance import inference_aec_2022_baselain_algorithm
from data.metrics import *
from aecmos import aecmos
epsilon = 1e-6

class Pl_module(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self,x):
        pred = self.model(x)
        return pred

# ======================================== main section ==================================================

Hydra_path = "/data/ssd/ofersc/NN_AEC_RLS/src/conf/"
@hydra.main(config_path=Hydra_path,config_name="test.yaml")
def main(args):
    # ============================= create instance model with regular Pl_module =============================================
    #sys.path.append(args.models_dir)
    # Model definition 
    model_type = GRU()
    model = Pl_module(model_type)
    model = model.load_from_checkpoint(model=model_type,checkpoint_path =args.infer_from_ckpt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Run model files
    #args.path_wav = '/data/ssd/ofersc/NN_AEC/data/external/Test/Synthetic/' # measurements=True
    args.path_wav = '/data/ssd/ofersc/datasets/Test/Synthetic/' # measurements=False
    
    #output_name = '_output_steered_rls.wav'
    output_name = '_output_rls.wav'
    #output_name = '_output_ideal_rls.wav'
    pred_model(args,model, output_name,save_output_files = True )
    
    option = '_output_rls.wav'
    #option = '_mic.wav'
    results_model_mean = Measurments(args,option)
    print(results_model_mean)

    option = '_linear_CV.wav'
    results_model_mean = Measurments(args,option)
    print(results_model_mean)

    results_echo_mos, results_deg_mos = aecmos(args,'output_steered_rls')
    results_echo_mos, results_deg_mos = aecmos(args,'output_rls')
    results_echo_mos, results_deg_mos = aecmos(args,'output_ideal_rls')
    results_echo_mos, results_deg_mos = aecmos(args,'linear') # From clearvox
    #results_echo_mos, results_deg_mos = aecmos(args,'cn')


    # Run AEC_2022 Baseline algorithm    
    # inference_aec_2022_baselain_algorithm(args)

if __name__ == '__main__':
    main()
    print('Done')

