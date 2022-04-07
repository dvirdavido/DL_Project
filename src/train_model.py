import os
import sys
sys.path.append('src/models/')
import hydra
import shutil
import warnings
import numpy as np
from models.def_model import *
from data.MyDataloader import *
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
warnings.filterwarnings("ignore", category=UserWarning)
from pytorch_lightning.plugins import DDPPlugin
np.set_printoptions(threshold=sys.maxsize)
import soundfile as sf
# import pysepm
# from data.metrics import *
# from aecmos import aecmos

# ======================== Model section ===========================


epsilon = 1e-12
class Pl_module(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model = GRU()
        self.criterion = nn.MSELoss()

    def forward(self,x):
        y_hat = self.model(x)     
        return y_hat

    def training_step(self, batch, batch_idx):
        echo, mic,InputToModel,ref = batch
        pred = self.model(InputToModel)
        loss = self.Loss(echo, pred , mic ,ref) 
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx): #in each end of epoch
        echo, mic, InputToModel,ref = batch
        pred = self.model(InputToModel)
        loss = self.Loss(echo, pred  ,mic,ref) 
        self.log('val_loss', loss)
        
    def test_step(self, batch, batch_idx): #in end of all training process
        echo, mic,InputToModel,ref = batch
        pred = self(InputToModel)
        loss = self.Loss(echo, pred ,mic,ref) 
        self.log('test_loss', loss)

    # def validation_epoch_end(self, outputs): #in each end of epoch
    #     ### real recordings
    #     if (np.mod(self.current_epoch,10)==0) & (self.current_epoch>0): # activate test after 10*n epochs n=1,2,3...
    #         self.args.path_wav = '/data/ssd/ofersc/datasets/Test/real/'
    #         results_mean = pred_model(self.args,self.model,save_output_files=True,measurements=False)
    #         # AECMOS for our outputs
    #         try: 
    #             results_echo_mos, results_deg_mos = aecmos(self.args,'_output_steered_rls')
    #             self.log("Echo_mos",results_echo_mos,rank_zero_only=True)
    #             self.log("Deg_mos",results_deg_mos,rank_zero_only=True)
    #         except: 
    #             print('AECMOS N/A')

    #         ### Synthetic recordings 
    #         self.args.path_wav = '/data/ssd/ofersc/datasets/Test/Synthetic/'
    #         results_mean = pred_model(self.args,self.model,save_output_files=True,measurements=True)

    #         self.log("LLR",results_mean['LLR'],rank_zero_only=True)
    #         self.log("CD",results_mean['CD'],rank_zero_only=True)
    #         self.log("aws_seg_snr",results_mean['aws_seg_snr'],rank_zero_only=True)
    #         self.log("PESQ",results_mean['PESQ'],rank_zero_only=True)
    #         self.log("ERLE",results_mean['ERLE'],rank_zero_only=True)

    #         print(results_mean)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer

    def Loss(self,echo,pred, mic,ref):
        # MMSE loss
        # loss = self.criterion(pred,target)
        # SNR loss
        #speech_est = pred[:,:,0:-1] * mic
        # RLS 
        #speech_est=0*mic
        #for n in range(len(speech_oracle[:,0,0])):
        #    speech_est[n,:,:] = RLS(torch.squeeze(pred[n,:,:]), torch.squeeze(mic[n,:,:]), torch.squeeze(ref[n,:,:]), self.args)
        
        echo_est=pred*mic

        #####
        echo = echo[:,50:,:] # Taking the echo as target !!!!!!!!!!!!!!
        echo_est = echo_est[:,50:,:]
        mic = mic[:,50:,:]
        echo_abs = torch.square(torch.abs(echo)) 
        others_abs = torch.square(torch.abs(mic-echo)) 
        flag_echo = torch.tensor( echo_abs > 1e-4 , dtype = torch.bool )
        flag_others = torch.tensor(  others_abs > 1e-4 , dtype = torch.bool )
        flag_dt = flag_others & flag_echo
        #Double talk loss: 
        Target_energy_dt = torch.zeros(len(echo[:,0,0])) ; Error_energy_dt = torch.zeros(len(echo[:,0,0]));
        Target_energy_st = torch.zeros(len(echo[:,0,0])) ; Error_energy_st = torch.zeros(len(echo[:,0,0]));

        for n in range(len(echo[:,0,0])):
            Target_energy_dt[n] = torch.sum(echo_abs[n, flag_dt[n,:,:] ])   
            Error_energy_dt[n]  = torch.sum(  torch.square( torch.abs( echo_est[ n,flag_dt[n,:,:] ] - echo[ n,flag_dt[n,:,:] ])))
            Target_energy_st[n] = torch.sum(echo_abs[n, flag_dt[n,:,:].logical_not() ])   
            Error_energy_st[n]  = torch.sum(  torch.square( torch.abs( echo_est[ n,flag_dt[n,:,:].logical_not() ] - echo[ n,flag_dt[n,:,:].logical_not() ])))
        
        alpha = 0.95
        Target_energy = alpha*Target_energy_dt + (1-alpha) * Target_energy_st
        Error_energy = alpha*Error_energy_dt + (1-alpha) * Error_energy_st
        loss = -10 * torch.mean( torch.log10(   torch.clamp( Target_energy ,min=epsilon  ) ) 
        - torch.log10(  torch.clamp(   Error_energy ,min=epsilon )   )  )  

        # if 0:
        #     #Visualize spectogram
        #     fig_0 = plt.figure()
        #     fig_0.suptitle('input')
        #     plt.imshow(flag_speech[0,].cpu().T,origin='lower',aspect='auto')
        #     plt.colorbar()
        #     plt.show()
        #     plt.savefig('/data/ssd/ofersc/NN_AEC/reports/figures/temp.png')
        #     plt.close()

        return loss 

# ======================================== main section ==================================================
Hydra_path = "C:\\Users\\sagitorr\\Documents\\University\\Final_Project\\Project\\dl_project\\src\\conf\\"
@hydra.main(config_path= Hydra_path,config_name="train.yaml")
def main(args):
# ====================== load config params from hydra ======================================
    # Storing the code in the model-folder
    pl_checkpoints_path = os.getcwd() + '\\'
    
    # save train_model.py and model_def.py files as part of hydra output
    shutil.copy(hydra.utils.get_original_cwd() + '\\dl_project\\src\\train_model.py', 'save_train_model.py')
    shutil.copy(hydra.utils.get_original_cwd() + '\\dl_project\\src\\models\\def_model.py', 'save_model_def.py')
    
    if args.debug_flag: # debug mode
        fast_dev_run=True
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    else: # training mode
        fast_dev_run=False
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

# ============================= main section =============================================

    model = Pl_module(args)
    # Managing the model storing by which loss, last loss, 5 best loss (minimums)...
    checkpoint_callback = ModelCheckpoint(
        monitor = args.ckpt_monitor,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_last = args.save_last,
        save_top_k = args.save_top_k,
        mode='min',
        verbose=True,
    )
    # Managing the convergence (stopping when there is no improvement)  
    earlystopping_callback = EarlyStopping(monitor = args.ckpt_monitor, 
                                            patience = args.patience)

    trainer = Trainer(  
                        # gpus=args.gpus, 
                        # accelerator = 'ddp',
                        # fast_dev_run=fast_dev_run, 
                        # check_val_every_n_epoch=args.check_val_every_n_epoch, 
                        # default_root_dir= pl_checkpoints_path,                       
                        # callbacks=[earlystopping_callback, checkpoint_callback], 
                        # log_gpu_memory=args.log_gpu_memory, 
                        # progress_bar_refresh_rate=args.progress_bar_refresh_rate,
                        # precision=32,
                        # plugins=DDPPlugin(find_unused_parameters=False),
                        # num_sanity_val_steps = 0
                     )

    data_module = StftDataModule(args)
    trainer.fit(model = model, datamodule = data_module)
    trainer.test(model = model)  
    checkpoint_callback.best_model_path

if __name__ == '__main__':
    main()

# Tensor Board:
# tensorboard --logdir models


# enter exist tmux session 0 :  tmux a -t 0
# open new session: tmux  
# ctrl-b and then w : see the windows
# choose windiow 
# ctrl-b and then , : rename the name of the window
# ctrl-b and then c : open new window 
# ctrl-b and then d : exit
# ctrl-d kill session
# ctrl-b and then x : kill window

# ofersc@ubuntu-beast:/data/ssd/ofersc/NNֹ_AEC$ tensorboard --logdir models/

#GPU status: 
# gpustat -i 5

# test


# connection problems: 
# In Moba: 
# ps aux | grep ofer
# kill -9 14407 # kill each former ssh connections




