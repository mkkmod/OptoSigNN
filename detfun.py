from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import scipy.signal as signal

# from torchaudio.functional import *
import matplotlib.pyplot as plt
import inspect

def lfilter_bg(waveform, a_coeffs, b_coeffs):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    klon lfilter() z torchaudio.functional
    
    waveform - dowolna liczba wymiarów, ale ostatni wymiar to czas
    
    """

    dim = waveform.dim()

    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    assert(a_coeffs.size(0) == b_coeffs.size(0))
    assert(len(waveform.size()) == 2)
    #assert(len(waveform.size()) == 1)
    assert(waveform.device == a_coeffs.device)
    assert(b_coeffs.device == a_coeffs.device)

    device = waveform.device
    dtype = waveform.dtype
    n_channel, n_sample = waveform.size()
    # n_sample = len(waveform)
    n_order = a_coeffs.size(0)
    assert(n_order > 0)

    # Pad the input and create output
    padded_waveform = torch.zeros(n_channel, n_sample + n_order - 1, dtype=dtype, device=device)
    padded_waveform[:, (n_order - 1):] = waveform
    padded_output_waveform = torch.zeros(n_channel, n_sample + n_order - 1, dtype=dtype, device=device)

    # Set up the coefficients matrix
    # Flip order, repeat, and transpose
    if False:
        a_coeffs_filled = a_coeffs.flip(0).repeat(n_channel, 1).t()
        b_coeffs_filled = b_coeffs.flip(0).repeat(n_channel, 1).t()

    #return waveform[0] * (torch.diag(a_coeffs_filled).sum()+torch.diag(b_coeffs_filled).sum())

    # Set up a few other utilities
    a0_repeated = torch.ones(n_channel, dtype=dtype, device=device) * a_coeffs[0]
    ones = torch.ones(n_channel, n_sample, dtype=dtype, device=device)
 
    # prevmem = torch.zeros(n_channel, n_order, dtype=dtype, device=device)
     
    for i_sample in range(n_sample):
 
        o0 = torch.zeros(n_channel, dtype=dtype, device=device)
  
        windowed_input_signal = padded_waveform[:, i_sample:(i_sample + n_order)]
        windowed_output_signal = padded_output_waveform[:, i_sample:(i_sample + n_order)].clone()
        
        # for ii in range(n_order - 1):
        #     prevmem[:, ii] = prevmem[:, ii + 1]
        # # bo windowed_output_signal "wybiega" o 1 próbkę do przodu
        # prevmem[:, -1] = 0
        # if False:
        #     assert(all((prevmem == padded_output_waveform[:, i_sample:(i_sample + n_order)]).tolist()))
 
        # o0.add_(torch.diag(torch.mm(windowed_input_signal, b_coeffs_filled)))
        o0.add_((windowed_input_signal * b_coeffs.flip(0)).sum(axis=1))
        
        # bad !!!!, ok!!!
        # o0.sub_(torch.diag(torch.mm(windowed_output_signal, a_coeffs_filled)))
        o0.sub_((windowed_output_signal * a_coeffs.flip(0)).sum(axis=1))
        
        if False:
            torch.max(torch.abs((windowed_input_signal*b_coeffs.flip(0)).sum(axis=1)-torch.diag(torch.mm(windowed_input_signal, b_coeffs_filled))))
        

        o0.div_(a0_repeated)
 
        padded_output_waveform[:, i_sample + n_order - 1] = o0
        
        #prevmem[:, -1] = o0
 
    # output = torch.min( ones, torch.max(ones * -1, padded_output_waveform[:, (n_order - 1):]))
    output = padded_output_waveform[:, (n_order - 1):]
     
    # unpack batch
    output = output.reshape(shape[:-1] + output.shape[-1:])
    return output     
     
    #return output[0]


def lowpass_biquad_gd(waveform, sample_rate, cutoff_freq, Q=0.707):
    # type: (Tensor, int, float, float) -> Tensor
    """  
    klon lowpass_biquad() z torchaudio.functional
    """
 
    GAIN = 1.
    w0 = 2 * math.pi * cutoff_freq / sample_rate
    # A = math.exp(GAIN / 40.0 * math.log(10))
    alpha = torch.sin(w0) / 2 / Q
    mult = math.exp(max(GAIN, 0) * math.log(10) / 20.0)
 
    b0 = (1 - torch.cos(w0)) / 2
    b1 = 1 - torch.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha
    return mult * lfilter_bg(
                waveform,
                torch.stack([a0, a1, a2]),
                torch.stack([b0, b1, b2])
            )


def highpass_biquad_gd(waveform, sample_rate, cutoff_freq, Q=0.707):
    # type: (Tensor, int, float, float) -> Tensor
    """  
    klon highpass_biquad() z torchaudio.functional
    """

    GAIN = 1.
    w0 = 2 * math.pi * cutoff_freq / sample_rate
    # A = math.exp(GAIN / 40.0 * math.log(10))
    alpha = torch.sin(w0) / 2. / Q
    mult = math.exp(max(GAIN, 0) * math.log(10) / 20.0)

    b0 = (1 + torch.cos(w0)) / 2
    b1 = -1 - torch.cos(w0)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * torch.cos(w0)
    a2 = 1 - alpha
    return mult * lfilter_bg(
                waveform,
                torch.stack([a0, a1, a2]),
                torch.stack([b0, b1, b2])
            )
    # return waveform * (a0 * a1 * a2 + b0 * b1 * b2)
    # return waveform * (torch.stack([a0, a1, a2]).sum() + torch.stack([b0, b1, b2]).sum())

def sosfilt_gd(x, sos):
    outsig = x
    for sect in sos:
        # sect = torch.Tensor(sect)
        outsig = lfilter_bg(outsig, sect[3:], sect[:3])
    return outsig


class SosModel(nn.Module):
    
    def __init__(self, sos):
        super(SosModel, self).__init__()
        if not isinstance(sos, torch.Tensor):
            sos = torch.Tensor(sos)
        self.sos = Parameter(sos.clone())

    def forward(self, x):
        # x dim: [batch, seq, channels]
        ftx = sosfilt_gd(x.transpose(1, 2), self.sos)
        ftx = ftx.transpose(1, 2)
        assert(ftx.size() == x.size())
        return ftx


class DetFun(nn.Module):
    
    def __init__(self, fsample):
        super(DetFun, self).__init__()
        
        if False:
            self.fsample = fsample
            self.bp = Parameter(torch.tensor([6.0, 20.0], requires_grad=True))
        
        if True:
            bp_par = torch.Tensor([6.0, 20.0]) / (fsample / 2)
            bp_sos = signal.butter(2, bp_par.tolist(), btype = "pass", output="sos")
            
            # bp_sos = Parameter(bp_sos)
            # self.bp_sos = bp_sos
            
            self.bp_sos = SosModel(bp_sos)
            self.lp_sos = SosModel(signal.butter(4, [10.0 / (fsample / 2)], btype = "low", output="sos"))
            
            # self.bp_sos = SosModel(torch.randn(2, 6))
            # self.lp_sos = SosModel(torch.randn(2, 6))

    def forward(self, x):
        # x torch.Size([8, 2048, 9])
        
        if False:
            Q = 0.1
            bp1 = torch.max(torch.tensor(1.0), self.bp[0])
            bp2 = torch.max(bp1 + 2.0, self.bp[1])

            ftx = highpass_biquad_gd(x.transpose(1, 2), self.fsample, bp1, Q=Q)
            ftx = highpass_biquad_gd(ftx, self.fsample, bp2, Q=Q)
            ftx = ftx.transpose(1, 2)
        
        if True:
            # ftx = sosfilt_gd(x.transpose(1, 2), self.bp_sos)
            # ftx = ftx.transpose(1, 2)

            ftx = self.bp_sos(x)
            ftx = self.lp_sos(ftx**2)
            
        assert(ftx.size() == x.size())
        return ftx

