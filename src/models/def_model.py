from functools import partial
import torch
from torch import nn

    
class GRU(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, dropout=0., win_len=512, context=1):
        super(GRU,self).__init__()
        freqs = (win_len // 2 + 1)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.context = context

        self.time_rnn = nn.GRU(freqs*(context+1), hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.time_fc1 = nn.Linear(hidden_size, hidden_size)
        #self.time_fc2 = nn.Linear(hidden_size, freqs + 1 )
        self.time_fc2 = nn.Linear(hidden_size, freqs  )
        self.activation = nn.Sigmoid()

    def forward(self, x, states=None):
        """
        Args:
            x: [B, T, F, 2*C] - mic & ref
            states (tuple): each [num_layers, B, hidden]
        """
        B, T, F, C = x.shape
        if states is None:
            states = x.new_zeros((self.num_layers, B, self.hidden_size))
            
        x = x.reshape(B, T, -1)  # [B, T, F*C]
        time_rnn_out, states = self.time_rnn(x, states)  # [B, T, H]
        
        out = self.time_fc1(time_rnn_out)  # [B, T, F]
        out = torch.relu(out)
        out = self.time_fc2(out)
        mask = self.activation(out)

        return mask 
       
