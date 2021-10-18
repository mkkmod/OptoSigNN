from torch import nn
import torch
from TCN.tcn import TemporalConvNet
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

# from torchaudio.functional import *
import matplotlib.pyplot as plt

from detfun import DetFun


class TCN(nn.Module):
    
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, fin_dropout = 0, useDetFun = False, fsample = 0):
        super(TCN, self).__init__()
        self._useDetFun = useDetFun
        if self._useDetFun:
            self.detfun = DetFun(fsample)
            
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        #self.drop = nn.Dropout(dropout)
        self.fin_drop = nn.Dropout(fin_dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        
        tol = 1e-30

        if self._useDetFun:
            ftx = self.detfun(x)
            
            # teraz prosta normalizacja 
            # TODO docelowo norm. online z kwantylami
            if False:
                fmx = ftx.max(axis=1).values
                fmx = torch.where(torch.abs(fmx) > tol, fmx, torch.ones(fmx.size()))
                
                fshape = ftx.size()
                # repfmx = fmx.unsqueeze(1).repeat(1, fshape[1], 1)    
                repfmx = fmx.unsqueeze(1).expand(fshape)       
                assert(repfmx.size() == fshape)
                
                ftx /= repfmx
                
        else:
            ftx = x
            
        ii = -1
        if False:
            print(self.detfun._bp_sos)

            ii += 1
            plt.plot(x[0, :, ii])
            plt.plot(ftx.detach().numpy()[0, :, ii])

            
        output = self.tcn(ftx.transpose(1, 2)).transpose(1, 2)
        output = self.fin_drop(output)
        
        # .double() to gryzie się z microsoft ML ONNX
        # output = self.linear(output).double()
        output = self.linear(output)
        
        # torch.Size([8, 2048, 1])
        return self.sig(output)
        return output


class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_dim, output_size=1, num_layers=2, dropout = 0, useDetFun = False, fsample = 0):
        super(LSTM, self).__init__()
        
        self._useDetFun = useDetFun
        if self._useDetFun:
            self.detfun = DetFun(fsample)
        
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers)
        self.linear = nn.Linear(hidden_dim, output_size)
        self.drop = nn.Dropout(dropout)

        # self.sig = nn.Sigmoid()

    def forward(self, x):
        
        # input of shape (seq_len, batch, input_size)
        if self._useDetFun:
            ftx = self.detfun(x)
        else:
            ftx = x
        
        ii = -1
        
        if False:
            print(self.detfun._bp_sos)

            ii += 1
            plt.plot(x[0, :, ii])
            plt.plot(ftx.detach().numpy()[0, :, ii])
  
        output, _ = self.lstm(ftx.transpose(0, 1))
        output = self.drop(output)
        output = self.linear(output).transpose(0, 1).double()
        # return self.sig(output)
        return output



class DUAL(nn.Module):
    
    def __init__(self, input_size, output_size, tcn_num_channels, tcn_kernel_size, tcn_dropout, tcn_fin_dropout, lstm_hidden_dim, lstm_num_layers, lstm_dropout=0):
        super(DUAL, self).__init__()
        
        self.tcn = TCN(input_size, output_size, tcn_num_channels, tcn_kernel_size, tcn_dropout, tcn_fin_dropout)
        self.lstm = LSTM(input_size, lstm_hidden_dim, output_size, lstm_num_layers, lstm_dropout)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        a = self.tcn(x)
        b = self.lstm(x)
        # return a * b
        return self.sig(a * b)
        




class SimpSos(nn.Module):
    
    def __init__(self, input_size, output_size, fin_dropout, fsample):
        super(SimpSos, self).__init__()

        self.detfun = DetFun(fsample)
            
        self.fin_drop = nn.Dropout(fin_dropout)
        self.linear = nn.Linear(input_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN

        ftx = self.detfun(x)
            
        ii = -1
        if False:
            print(self.detfun._bp_sos)

            ii += 1
            plt.plot(x[0, :, ii])
            plt.plot(ftx.detach().numpy()[0, :, ii])
            
        output = self.fin_drop(ftx)
        output = self.linear(output).double()
        # torch.Size([8, 2048, 1])
        return self.sig(output)
        return output


