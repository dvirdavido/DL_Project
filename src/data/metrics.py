import numpy as np
import pysepm
import os
import torch
from scipy.io.wavfile import write
import sys
import soundfile
import matplotlib.pyplot as plt

#sys.path.append('/workspace/inputs/ayal/dereverberation/src/spec2spec/data/')
from data.data_utils import AudioPreProcessing


def post_process(pred,mic, ref, args):
    # Filtering
    
    beta = 6
    output = torch.exp((torch.squeeze(pred)-1)*beta) * mic
    ## ISTFT
    #output_time = torch.istft(torch.swapaxes(output, 0, 1),window=torch.hamming_window(args.window_size, device=pred.device),hop_length=args.overlap, n_fft=args.window_size,return_complex=False,center=False)

    return output

def RLS(pred, mic, ref, args):

    device = mic.device
    Taps = 4
    alpha = 0.95
    #beta = 1/(1-alpha)
    Phi_ref = torch.zeros(len(mic[0,:]),Taps,Taps,len(mic[:,0]) , dtype=torch.complex128).to(device)
   
    Phi_mic_ref = torch.zeros(len(mic[0,:]),Taps,len(mic[:,0]),dtype=torch.complex128).to(device)
    #C = torch.zeros(len(mic[0,:]),Taps,dtype=torch.complex128).to(device)
    G = torch.zeros(len(mic[0,:]),Taps,len(mic[:,0]), dtype=torch.complex128).to(device)
    ref_buffer = torch.zeros(len(mic[0,:]),len(mic[:,0]),Taps,dtype=torch.complex128).to(device)
    DL = torch.eye(Taps,dtype=torch.complex128).to(device) * 1e-3
    DL = DL.repeat(len(mic[0,:]),1,1)
    output = torch.zeros(len(mic[:,0]),len(mic[0,:]),dtype=torch.complex128).to(device)
    #dim = 0 ; sign = 1
    #add_vec = 0*ref_buffer
    #En_ref = torch.sum(torch.abs(ref),1) 
    #mic_echo = torch.split(pred * mic,1)
    mic_echo = torch.split(mic,1)
    alpha_ = torch.zeros(len(mic[0,:]),1,dtype=torch.complex128).to(device)
    
    for m in range(len(mic[:,0])): # loop on time
        if m>0:
            ref_buffer[:,m,0:-1] = ref_buffer[:,m-1,1:]
            ref_buffer[:,m,Taps-1] = ref[m,:]
            output[m,:] = mic[m,:] - torch.squeeze(torch.matmul( torch.unsqueeze(ref_buffer[:,m,:],1),torch.unsqueeze(G[:,:,m-1],-1)))
        else:
            ref_buffer[:,m,Taps-1] = ref[m,:]
            output[m,:] = mic[m,:] 

        # #if 1: #En_ref[m] > 1:
        #     ## Update G
        # add_vec = 0 * ref_buffer
        # add_vec[:,0,dim] = 0.1 * sign
        # dim = (dim+1) % Taps
        # sign = -sign

        # B = 1/alpha * Phi_ref
        # C = torch.matmul(B, torch.conj(torch.transpose(ref_buffer + add_vec ,1,2))) ; 
        # d = torch.matmul(ref_buffer +add_vec , C)
        # d = d.real + 0*1j
        # d = 1/(beta + d)
        # Phi_ref = B - torch.matmul( torch.matmul(C,d) , torch.transpose(torch.conj(C),1,2)) 
        # #Phi_ref = 0.5*(Phi_ref + torch.conj(torch.transpose(Phi_ref,1,2) )) # ensure hermitian matrix

        # E = torch.matmul(Phi_ref, torch.transpose(torch.conj(ref_buffer+add_vec ),1,2))
        # G = G + (1-alpha) *  torch.matmul( E ,  torch.unsqueeze(torch.unsqueeze(output[m,:],-1),-1)   )
        
        ##### RLS without matrix inversion lemma
        alpha_ = torch.where(pred[m,:] < 0.5 , 1.0 , 0.99)
        if m>0:
            alpha = alpha_.unsqueeze(1).repeat(1,Taps)
            Phi_mic_ref[:,:,m] = alpha * Phi_mic_ref[:,:,m-1] + (1-alpha) * torch.squeeze(torch.matmul( torch.conj(torch.transpose(torch.unsqueeze(ref_buffer[:,m,:],1),1,2)) , torch.unsqueeze(torch.unsqueeze(mic_echo[m],-1),-1) ))
            alpha = alpha_.unsqueeze(1).unsqueeze(1).repeat(1,Taps,Taps)
            Phi_ref[:,:,:,m] = alpha * Phi_ref[:,:,:,m-1] + (1-alpha) * torch.matmul(torch.conj(torch.transpose(torch.unsqueeze(ref_buffer[:,m,:],1),1,2)) , torch.unsqueeze(ref_buffer[:,m,:],1)   )
        else:
            Phi_mic_ref[:,:,m] =  torch.squeeze(torch.matmul( torch.conj(torch.transpose(torch.unsqueeze(ref_buffer[:,m,:],1),1,2)) , torch.unsqueeze(torch.unsqueeze(mic_echo[m],-1),-1) ))
            Phi_ref[:,:,:,m] = torch.matmul(torch.conj(torch.transpose(torch.unsqueeze(ref_buffer[:,m,:],1),1,2)) , torch.unsqueeze(ref_buffer[:,m,:],1)   )

        G[:,:,m] = torch.squeeze(torch.matmul(torch.linalg.inv(  Phi_ref[:,:,:,m] + DL),  torch.unsqueeze(Phi_mic_ref[:,:,m],-1) ) )
 
    #output_time = torch.istft(torch.swapaxes(output, 0, 1),window=torch.hamming_window(args.window_size, device=mic.device),hop_length=args.overlap, n_fft=args.window_size,return_complex=False,center=False)

    return output

#########    NUMPY ###################################
    # device = mic.device
    # Taps = 4
    # alpha = 0.99
    # beta = 1/(1-alpha)
    # #Phi_ref = torch.eye(Taps,dtype=np.complex128).to(device) * 100
    # Phi_ref = np.eye(Taps,dtype=np.complex128) * 100
    # Phi_ref = np.repeat(np.expand_dims(Phi_ref,0), len(mic[0,:]),0)
    # Phi_mic_ref = np.zeros((len(mic[0,:]),Taps,1),dtype=np.complex128)
    # C = np.zeros((len(mic[0,:]),Taps),dtype=np.complex128)
    # G = np.zeros((len(mic[0,:]),Taps),dtype=np.complex128)
    # ref_buffer = np.zeros((len(mic[0,:]),Taps),dtype=np.complex128)
    # DL = np.eye(Taps,dtype=np.complex128) * 1e-6
    # DL = np.repeat(np.expand_dims(DL,0), len(mic[0,:]),0)

    # mic = mic.cpu().numpy()
    # ref = ref.cpu().numpy()
    # output = 0*mic
    
    # dim = 0 ; sign = 1
    # add_vec = 0*ref_buffer
    # En_ref = np.sum(np.abs(ref),1) 

    # for m in range(len(mic[:,0])): # loop on time

    #     ref_buffer[:,0:-1] = ref_buffer[:,1:]; ref_buffer[:,Taps-1] = ref[m,:]
    #     output[m,:] = mic[m,:] - np.sum(np.multiply(np.squeeze(G), ref_buffer),1)

    #     #if 1: #En_ref[m] > 1:
    #         ## Update G
    #         # add_vec = 0 * ref_buffer
    #         # add_vec[:,dim] = 0.0001 * sign
    #         # dim = (dim+1) % Taps
    #         # sign = -sign

    #         # B = 1/alpha * Phi_ref
    #         # C = torch.matmul(B, torch.unsqueeze(ref_buffer + add_vec,-1)) ; 
    #         # d = torch.sum(torch.mul(torch.conj(ref_buffer + add_vec), torch.squeeze(C)),1)
    #         # d = d.real + 0*1j
    #         # d = 1/(beta + d)
    #         # Phi_ref = B - torch.matmul( torch.matmul(C,torch.unsqueeze(torch.unsqueeze(d,-1),-1)) , torch.unsqueeze(torch.squeeze(torch.conj(C)),1)) 
    #         # Phi_ref = 0.5*(Phi_ref + torch.conj(torch.transpose(Phi_ref,1,2) )) # ensure hermitian matrix

    #         # E = torch.matmul(Phi_ref, torch.unsqueeze(torch.conj(ref_buffer + add_vec),-1))
    #         # G = G + (1-alpha) * torch.squeeze( torch.matmul( E ,  torch.unsqueeze(torch.unsqueeze(output[m,:],-1),-1) ) )

    #     Phi_ref = alpha * Phi_ref + (1-alpha) * np.matmul(np.expand_dims(ref_buffer,-1) , np.conj(np.expand_dims(ref_buffer,1)))
    #     #Phi_ref = 0.5*(Phi_ref + torch.conj(torch.transpose(Phi_ref,1,2) )) # ensure hermitian matrix

    #     Phi_mic_ref = alpha * Phi_mic_ref + (1-alpha) * np.matmul(  np.expand_dims(np.conj(ref_buffer),-1) , np.expand_dims(np.expand_dims(mic[m,:],-1),-1) )
    #     G = np.linalg.solve(  Phi_ref + DL , Phi_mic_ref ) 
    #     #np.mean(np.matmul(Phi_ref + DL,G) - Phi_mic_ref) 
            
    # output_time = torch.istft(torch.swapaxes(torch.from_numpy(output).to(device), 0, 1),window=torch.hamming_window(args.window_size, device=device),hop_length=args.overlap, n_fft=args.window_size,return_complex=False,center=False)



def tests(args,output,mic,speech,results_model):

    mic = np.array(np.squeeze(mic))+1e-10
    speech = np.array(np.squeeze(speech))+1e-10
    if torch.is_tensor(output):
        output = np.array(np.squeeze(output.cpu()))
    output = np.pad(output, (0, len(mic)-len(output)), 'constant', constant_values=(0, 0)) + 1e-10

    # PESQ
    speech_padded = np.convolve( np.abs(speech) , np.ones(1000)/1000 ,mode='same' )
    output_ = np.where( speech_padded > 1e-8 , output, 1e-10)
    mic_ = np.where( speech_padded > 1e-8 , mic, 1e-10)
    Pesq_improvement = pysepm.pesq(speech, output_, args.sample_rate)[1] - pysepm.pesq(speech, mic_, args.sample_rate)[1]
    results_model["PESQ"].append(Pesq_improvement)

    if 0:
        fig_0 = plt.figure()
        fig_0.suptitle('input')
        plt.plot(STD.cpu(),alpha=0.5); 
        plt.show()
        plt.savefig('/data/ssd/ofersc/NN_AEC/reports/figures/temp.png')
        plt.close()
        
    # ERLE
    speech_padded = np.convolve( np.abs(speech) , np.ones(1000)/1000 ,mode='same' )
    #plt.plot(speech_padded) 
    output_ = np.where( speech_padded < 1e-8 , output, 1e-10)
    mic_ = np.where( speech_padded < 1e-8 , mic, 1e-10)
    ERLE =  10*(np.log10(np.sum(np.square(output_))) - np.log10(np.sum(np.square(mic_))))
    results_model["ERLE"].append(ERLE)
    
    results_model["LLR"].append(pysepm.llr(speech, output, args.sample_rate))
    results_model["CD"].append(pysepm.cepstrum_distance(speech, output, args.sample_rate))
    results_model["aws_seg_snr"].append(pysepm.fwSNRseg(speech, output, args.sample_rate))

    return results_model

def inference(args,audio_pre_process,model,mic_wav,ref_wav):

    #device = torch.device('cuda:{:d}'.format(next(model.parameters()).device.index)) if torch.cuda.is_available() else 'cpu'

    echo, mic,input_data, ref  = audio_pre_process.transformations(mic_wav,ref_wav,0,0)
    if torch.cuda.is_available():
        input_data = input_data.to(torch.device('cuda'))
        mic = mic.to(torch.device('cuda'))

    pred = model(torch.unsqueeze(input_data, 0))
    #pred = pred.cpu().numpy()
    #pred = pred.squeeze()
    
    # Oracle pred ################################################
    if 0:
        echo_wav = mic_wav[0:-8] + '_echo.wav'
        echo,_ = soundfile.read(echo_wav)
        if len(echo) < 160000:
            echo = np.pad(echo, (0, 160000-len(echo)), 'constant', constant_values=(0, 0))    
        Echo = torch.stft( torch.tensor(echo, device=pred.device), window=torch.hamming_window(args.window_size, device=pred.device), hop_length=args.overlap, n_fft=args.window_size,return_complex=True,center=False).T
        echo_abs = torch.square(torch.abs(Echo)) 
        others_abs = torch.square(torch.abs(mic-Echo)) 
        pred_oracle = echo_abs / (echo_abs + others_abs + 1e-10)
    ################################################################

    #pred_oracle = 
    output_hybrid = RLS( torch.squeeze(pred), mic, ref, args)
    #output_hybrid = RLS( 1, mic, ref, args)
    #output_hybrid = RLS( pred_oracle, mic, ref, args)

    # ISTFT
    output_hybrid_time = torch.istft(torch.swapaxes(output_hybrid, 0, 1),window=torch.hamming_window(args.window_size, device=pred.device),hop_length=args.overlap, n_fft=args.window_size,return_complex=False,center=False)

    if 0:
        #Visualize spectogram
        fig_0 = plt.figure()
        fig_0.suptitle('input')
        plt.imshow(torch.squeeze(pred_oracle).cpu().T,origin='lower',aspect='auto')
        plt.colorbar()
        plt.show()
        plt.savefig('/data/ssd/ofersc/NN_AEC_RLS/reports/figures/temp.png')
        plt.close()

    return output_hybrid_time

def pred_model(args,model,output_name,save_output_files):
    model = model.eval()
    # filelist_mic = os.listdir(args.path_mic_wav)
    # filelist_ref = os.listdir(args.path_ref_wav)
    # filelist_speech = os.listdir(args.path_speech_wav)

    filelist = os.listdir(args.path_wav)
    filelist_mic = list()
    for file in filelist:
        if file.endswith('mic.wav'):
            filelist_mic.append(file)
    
    audio_pre_process = AudioPreProcessing(args)

    with torch.no_grad():
        for x in range(len(filelist_mic)):
            #print(x/len(filelist_mic))
            tmp = filelist_mic[x]
            mic_wav = args.path_wav + tmp
            ref_wav = args.path_wav + tmp[0:-8] + '_lpb.wav'

            output = inference(args,audio_pre_process,model,mic_wav,ref_wav)
            

            # save outputs
            if save_output_files:
                tmp=filelist_mic[x]
                output_file_generated = args.path_wav + tmp[0:-8] + output_name
                write(output_file_generated, args.sample_rate, (output*2**15).cpu().numpy().T.astype(np.int16)) 

                #output_file_generated = args.path_wav + tmp[0:-8] + '_output_regular_rls.wav'
                #write(output_file_generated, args.sample_rate, (output_rls*2**15).cpu().numpy().T.astype(np.int16)) 
                
    return 


def Measurments(args,option):
    # filelist_mic = os.listdir(args.path_mic_wav)
    # filelist_ref = os.listdir(args.path_ref_wav)
    # filelist_speech = os.listdir(args.path_speech_wav)

    filelist = os.listdir(args.path_wav)
    filelist_mic = list()
    for file in filelist:
        if file.endswith('mic.wav'):
            filelist_mic.append(file)

    results_model = {"LLR":[],"CD":[],"aws_seg_snr":[],"PESQ":[],"ERLE":[]}

    with torch.no_grad():
        for x in range(len(filelist_mic)):
            tmp = filelist_mic[x]
            mic_wav = args.path_wav + tmp
            ref_wav = args.path_wav + tmp[0:-8] + '_lpb.wav'

            
            # Measurements: 
            mic,_ = soundfile.read(mic_wav)
            speech_wav = args.path_wav + tmp[0:-8] + '_target.wav'
            speech,_ = soundfile.read(speech_wav)

            #output_wav = args.path_wav + tmp[0:-8] + 'linear_CV.wav'
            output_wav = args.path_wav + tmp[0:-8] + option
            output,_ = soundfile.read(output_wav)

            results_model = tests(args,output,mic,speech,results_model)

    results_model_mean=0
    results_model_mean = {k:sum(x)/len(x) for k,x in results_model.items()}

    return results_model_mean

def test_model(args,model):

    model = model.eval()
    #os.makedirs(os.getcwd()+'/test_results/', exist_ok=True)

    audio_pre_process = AudioPreProcessing(args)
    with torch.no_grad():

        output = inference(args,audio_pre_process,model, args.mic_wav, args.ref_wav)
        tmp = args.mic_wav
        output_file_generated =  tmp[0:-8] + '_output.wav'
        #output_file_generated = args.Output_saving_directory_single_test + args.mic_wav.replace('_mic.wav','_output.wav')
        write(output_file_generated, args.sample_rate, (output*2**15).cpu().numpy().T.astype(np.int16)) 
        
    return
