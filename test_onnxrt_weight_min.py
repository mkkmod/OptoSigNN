'''
Created on 5 mar 2020

@author: m.krej
'''

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import onnxruntime as rt
import numpy as np

import matplotlib.pyplot as plt

class BadBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, dontbebad=False):
        super(BadBlock, self).__init__()
        conv = nn.Conv1d(n_inputs, n_outputs, kernel_size)
        self.conv = conv if dontbebad else weight_norm(conv, dim=None) 
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv, self.relu)

   
    def forward(self, x):
        return self.net(x)


savname_onnx = 'test_onnxrt_model.onnx'
n_inp = 8
seq_len = 256
ksize = 6
n_channels = 10 

dontbebad = True
dontbebad = False

torch_model = BadBlock(n_inp, n_channels, ksize, dontbebad)

inp_shape = [1, n_inp, seq_len]
dummy_input = torch.randn(inp_shape)
real_input = torch.randn(inp_shape)

torch.onnx.export(torch_model, dummy_input, savname_onnx)

ref_out = torch_model(real_input).detach().numpy()
sess = rt.InferenceSession(savname_onnx)
input_name = sess.get_inputs()[0].name
result = sess.run(None, {input_name: real_input.numpy()})
result = np.array(result)

ref_out = ref_out[0, 0, :] 
result = result[0, 0, 0, :]

mse = ((ref_out - result) ** 2).mean()

print("MSE = {:.10f}".format(mse))
print("PASSED" if mse < 1e-3 else "FAILED")

if False:
    plt.plot(ref_out)
    plt.plot(result)
    plt.show()

