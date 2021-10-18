import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import onnx
import onnxruntime as rt
import numpy as np

import sys
sys.path.append("../TCN")
from TCN.tcn import *              
from model import *
              
print(torch.__version__)
print(onnx.__version__)


savname_onnx = "learnTCN_model.onnx"
savname_onnx = 'test_onnxrt_model.onnx'




class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x


class SigmaMd(nn.Module):
    def __init__(self, inp_size, out_size):
        super(SigmaMd, self).__init__()
         
        self.linear = nn.Linear(inp_size, out_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # .double() to ZŁO
        # outp = self.linear(x).double()
        outp = self.linear(x)
        return self.sig(outp)


class BadBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, dontbebad=False):
        super(BadBlock, self).__init__()
        conv = nn.Conv1d(n_inputs, n_outputs, kernel_size)
        self.conv = conv if dontbebad else weight_norm(conv) 
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv, self.relu)

    def forward(self, x):
        return self.net(x)


torch_model = TestModel()

torch_model = nn.Sigmoid()
dummy_input = torch.tensor(np.arange(100.0)).float()

n_inp = 9
seq_len = 256
ksize = 6


if False:
    n_channels = [1] * 5 
    n_channels = [2] * 1 

    torch_model = TCN(n_inp, 1, n_channels, ksize, 0)
    inp_shape = [1, seq_len, n_inp]

if False:
    n_channels = [1] * 5 
    n_channels = [2] * 1 

    torch_model = TemporalConvNet(n_inp, n_channels, ksize)
    inp_shape = [1, n_inp, seq_len]

if False:
    dilation_size = 2
    dilation_size = 4
    dilation_size = 1
    
    n_channels = 1
    n_channels = 10 
    torch_model = TemporalBlock(n_inp, n_channels, ksize, stride=1, dilation=dilation_size, padding=(ksize-1) * dilation_size)
    inp_shape = [1, n_inp, seq_len]

if True:
    n_channels = 1
    n_channels = 10 
    torch_model = BadBlock(n_inp, n_channels, ksize)
    inp_shape = [1, n_inp, seq_len]

if False:
    torch_model = SigmaMd(seq_len, seq_len)
    #dummy_input = torch.tensor(np.arange(seq_len)).float()
    #dummy_input = torch.zeros(seq_len)
    # real_input = torch.randn(seq_len)
    inp_shape = [seq_len]



dummy_input = torch.randn(inp_shape)
real_input = torch.randn(inp_shape)

_ = torch_model.eval()
ref_out = torch_model(real_input).detach().numpy()
torch.onnx.export(torch_model, dummy_input, savname_onnx, verbose=False, opset_version=11)
  
sess = rt.InferenceSession(savname_onnx)
input_name = sess.get_inputs()[0].name
result = sess.run(None, {input_name: real_input.numpy()})
result = np.array(result)

ref_out.shape   
result.shape

if np.argmax(result.shape) == 0:
    plt.plot(ref_out)
    plt.plot(result[0])
if np.argmax(result.shape) == 2:
    plt.plot(ref_out[0, :, 0])
    plt.plot(result[0, 0, :, 0])
if np.argmax(result.shape) == 3:    
    plt.plot(ref_out[0, 0, :])
    plt.plot(result[0, 0, 0, :])


if False:
    onnx_model = onnx.load(savname_onnx)
    print(onnx_model)
    onnx.checker.check_model(onnx_model)
    print(onnx_model.opset_import)

plt.show()

print("done")
