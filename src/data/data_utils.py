from re import M
import torch
import warnings
import matplotlib.pyplot as plt
import soundfile
warnings.filterwarnings("ignore", category=UserWarning)
import torchaudio
# torchaudio.set_audio_backend("sox_io")
import librosa
import numpy as np
from scipy.io.wavfile import write
# ==================== Hyper-Params =================================================================================
epsilon = 1e-12

def create_input(args,mic,ref):
    pad_before = args.context - 1
    ref = torch.nn.functional.pad(ref, [0, 0, pad_before, 0])
    # ref = ref.pad(ref, [0, 0, pad_before, 0])
    ref = ref.unfold(dimension=0, size=args.context, step=1)
    input_data = torch.cat((mic.unsqueeze(2),ref),axis = 2)
    return input_data


class AudioPreProcessing():
    def __init__(self,args, device=None):
        self.args = args
        self.device = device

    def transformations(self,mic_wav,ref_wav,target_wav,echo_wav):
        # load signal

        ref_wav, _ = soundfile.read(ref_wav)

        if target_wav==0: # INFERENCE ONLY
            target = 0
            mic_wav, _ = soundfile.read(mic_wav)

        else: 
            # rand speech
            amount = self.args.amount.train
            tmp = np.round(amount*np.random.uniform(low=0.0, high=1.0, size=None))
            if tmp==amount: 
                tmp=tmp-1
            tmp = tmp.astype('str')
            target_wav ='/data/ssd/ofersc/NN_AEC/data/AEC_challenge/nearend_speech_scaled/nearend_speech_fileid_' + tmp[0:-2]  + '.wav'   
            speech_wav,_ = soundfile.read(target_wav)
            
            # rand noise
            tmp = np.round(amount*np.random.uniform(low=0.0, high=1.0, size=None))
            if tmp==amount: 
                tmp=tmp-1
            tmp = tmp.astype('str')
            noise_wav ='/data/ssd/ofersc/NN_AEC/data/AEC_challenge/nearend_noise/nearend_noise_fileid_' + tmp[0:-2]  + '.wav'   
            
            noise_wav,_ = soundfile.read(noise_wav)
            if len(noise_wav) < len(speech_wav):
                noise_wav = np.pad(noise_wav, (0, len(speech_wav)-len(noise_wav)), 'constant', constant_values=(0, 0))
            
            echo_wav, _  = soundfile.read(echo_wav)
            if len(echo_wav) < len(speech_wav):
                echo_wav = np.pad(echo_wav, (0, len(speech_wav)-len(echo_wav)), 'constant', constant_values=(0, 0))
            
            # Build mic
            Speech_std = 0.02 * np.random.normal(0,1)  + 0.03
            phi_speech = np.mean(np.square(speech_wav))
            speech_wav = speech_wav * Speech_std / (np.sqrt(phi_speech) + 1e-12 )

            SER = 5 * np.random.normal(0,1)  + 0
            phi_echo = np.mean(np.square(echo_wav))
            gain = Speech_std * np.sqrt(10**(-SER/10)  / (phi_echo + 1e-12)  )
            
            echo = gain * echo_wav
            mic_wav = speech_wav + echo

            SNR = 10 * np.random.normal(0,1)  + 30
            phi_noise = np.mean(np.square(noise_wav))
            gain = Speech_std * np.sqrt(10**(-SNR/10) / (phi_noise + 1e-12) )
            
            mic_wav = mic_wav + gain * noise_wav[0:len(mic_wav)]

            # Set target
            target = torch.stft(torch.tensor(echo),window=torch.hamming_window(self.args.window_size),hop_length=self.args.overlap, n_fft=self.args.window_size,return_complex=True,center=False).T
            
        # STFT
        mic = torch.stft(torch.tensor(mic_wav),window=torch.hamming_window(self.args.window_size),hop_length=self.args.overlap, n_fft=self.args.window_size,return_complex=True,center=False).T
        ref = torch.stft(torch.tensor(ref_wav),window=torch.hamming_window(self.args.window_size),hop_length=self.args.overlap, n_fft=self.args.window_size,return_complex=True,center=False).T
        #
        # write('/data/ssd/ofersc/NN_AEC/reports/figures/mic.wav', 16000, (mic_wav*2**15).astype(np.int16)) 
        # write('/data/ssd/ofersc/NN_AEC/reports/figures/noise.wav', 16000, (gain * noise_wav[0:len(mic_wav)]*2**15).astype(np.int16)) 
        # write('/data/ssd/ofersc/NN_AEC/reports/figures/speech.wav', 16000, (speech_wav*2**15).astype(np.int16)) 
        # write('/data/ssd/ofersc/NN_AEC/reports/figures/echo.wav', 16000, (echo_wav*2**15).astype(np.int16)) 
        # write('/data/ssd/ofersc/NN_AEC/reports/figures/ref.wav', 16000, (ref_wav*2**15).astype(np.int16)) 

        #mic_ = torch.istft(mic.T,window=torch.hamming_window(self.args.window_size),hop_length=self.args.overlap, n_fft=self.args.window_size,return_complex=False,center=False)
        #plt.plot(mic_wav)
        #plt.plot(mic_wav)

        # Input of the nodel!
        input_spec = create_input(self.args,mic,ref)
        input_data =  torch.tensor(torch.abs(input_spec),dtype=torch.float)
        input_data = torch.log10(torch.clamp(input_data, min=epsilon))
        
        # Normalization:
        alpha = 0.997; M = 0; S = 1 #M = -1; S = 4 
        for n in range(len(input_data[:,0,0])):
            M = alpha*M + (1-alpha)*input_data[n,:,:]
            S = alpha*S + (1-alpha)*torch.square(torch.abs(input_data[n,:,:]))
            input_data[n,:,:] = (input_data[n,:,:] - M) / (torch.sqrt(S - torch.square(M))  + 1e-12)


        #Visualize signal
        # fig_0 = plt.figure()
        # fig_0.suptitle('input')
        # plt.plot(mic_wav)
        # plt.show()
        # plt.savefig('/data/ssd/ofersc/NN_AEC/reports/figures/temp.png')
        # plt.close()
        
        return target, mic, input_data ,ref