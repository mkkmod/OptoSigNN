import pyreadr
import os

import sys

# from matplotlib.pyplot import axis
sys.path.append("../TCN")
import torch
import torch.onnx

import torch.nn as nn
# from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math
from model import *

from myutils import *
from my_sync_loss import *
from my_tacho_loss import *


import glob
import random

# import subprocess
from getpass import getpass
import os

from locals import LocalMaxima

PRECOMP_TARG_TACHO = True
PRECOMP_TARG_TACHO = False

# 16s zapisy
MODE_16s = False
MODE_16s = True

USE_DET_FUN = True
USE_DET_FUN = False

pw = ''
if 0:
    # pw = input("pw? :") 
    pw = getpass() 

if not MODE_16s:
    baseDataPath = "/home/mkrej/dyskE/MojePrg/_R/OptoHrSrcSigQuality/ExampData/_online"
    siglen = 2 ** 13
else:
    baseDataPath = "/home/mkrej/dyskE/MojePrg/_R/OptoHrSrcSigQuality/ExampData/_online_16s"
    baseDataPath = "/home/mkrej/dyskE/MojePrg/_R/OptoHrSrcSigQuality/ExampData/_online_16s_v2"
    siglen = 2 ** 14
    
inpFiles = [f for f in glob.glob(baseDataPath + "/**/*.*") if f.lower().endswith('mx.rdata')]

allPaths = [(f, f.replace("_mx.Rdata", "_targ.Rdata").replace("_mx.RData", "_targ.RData")) for f in inpFiles]


for t in allPaths:
    for f in t:
        if not os.path.exists(f):
            raise Exception("FILE NOT EXISTS!!", f)
    # print(t, "\n")

down = 4
down = 8
fsample = 1000

flt_param = [6, 20] if not USE_DET_FUN else None 
flt_param = [3, 30] if not USE_DET_FUN else [1, 30] 
flt_param = [6, 20] if not USE_DET_FUN else [1, 30] 

print("loader filter (Hz) = ", flt_param)

all_inputs, all_targets = load_data(allPaths, fsample, down, siglen, filter=flt_param)

fsample = fsample / down

print("all_inputs len: ", len(all_inputs))

all_inputs.shape
all_targets.shape

if False:
    plt.plot(all_targets[3, :], 'r-')

all_inputs = torch.Tensor(all_inputs.astype(np.float64))
# all_targets = torch.Tensor(all_targets.astype(np.float64))
all_targets = torch.from_numpy(all_targets.astype(np.float64))

ind = list(range(len(all_inputs)))

random.shuffle(ind)

# idiv = int(random.random() * len(ind))

# Validation fraction
div_fract = 0.2
 
clip = 0.1
lrate = 0.0005
dropout = .25

idiv = int(len(ind) * div_fract)

batch_size = min(32 , idiv)

if False:
    print("\n*** dbg batch_size !!!\n")
    batch_size = 1

print("batch_size = ", batch_size)

wndlen2 = int(fsample * 60 / 240 * .8)
hr_interv = int(0.05 * fsample)
valthresh = 0.1

itest = ind[:idiv]
itrain = ind[idiv:]

if False and MODE_16s:
    # 2020 05 27     
    itest  =  [58, 109, 94, 31, 21, 138, 92, 102, 136, 67, 8, 25, 74, 89, 107, 45, 77, 66, 75, 148, 129, 53, 133, 87, 50, 19, 32, 135, 152, 103]
    itrain =  [139, 72, 39, 85, 128, 71, 1, 11, 56, 88, 2, 105, 83, 132, 44, 130, 144, 12, 41, 126, 149, 30, 73, 137, 10, 68, 147, 106, 127, 35, 131, 120, 82, 18, 34, 37, 20, 4, 90, 123, 48, 115, 15, 91, 23, 55, 22, 114, 6, 57, 26, 95, 125, 124, 29, 118, 65, 52, 13, 60, 143, 5, 79, 43, 38, 122, 98, 117, 70, 46, 28, 36, 97, 69, 146, 116, 61, 150, 108, 140, 96, 142, 104, 111, 9, 16, 17, 3, 151, 113, 33, 134, 86, 81, 7, 54, 84, 14, 59, 145, 24, 76, 119, 0, 63, 121, 141, 62, 40, 112, 100, 110, 99, 93, 27, 80, 49, 47, 64, 78, 42, 101, 51]


    if len(ind) < len(itest) + len(itrain):
        raise Exception("Fixed train/test mismatch!!")
    
    if len(ind) > len(itest) + len(itrain):
        print("\n***Fixed train/test set is smaller than avail. set !!")



print("itest  = ", itest)
print("itrain = ", itrain)

if False and len(itest) + len(itrain) != len(all_inputs):
    raise Exception("code ex") 

# test_input = all_inputs[-1]
# test_target = all_targets[-1] 
#  
# train_inputs = all_inputs[0:-1] 
# train_targets = all_targets[0:-1]

test_inputs = all_inputs[itest]
test_targets = all_targets[itest] 
 
train_inputs = all_inputs[itrain] 
train_targets = all_targets[itrain]

test_inputs.shape
test_targets.shape

train_inputs.shape
train_targets.shape

# nhid - liczba 'filtrów' w warstwach ukrytych, 
# w warstwie wejściowej liczba kanałów, w warstwie końcowej 1 wyjście

ksize = 6

n_channels = [16] * 6 
     
sync_time = .23  


idl_div = 2 if MODE_16s else 4              
idl_div = 3 if MODE_16s else 4
idl_div = 4

levels = len(n_channels)

max_sig_num = 9
fin_dropout = .13

# zasięg patrzenia do tyłu [s] : 
print("wnd :", 2 ** (levels - 1) * ksize / fsample)
print("ksize = {} channels = {}".format(ksize, n_channels))
print("lrate = {} ".format(lrate))

if True:
    
    if 1:
        print("model TCN")
        model = TCN(
            input_size=max_sig_num,
            output_size=1,
            num_channels=n_channels,
            kernel_size=ksize,
            dropout=dropout,
            fin_dropout=fin_dropout,
            useDetFun=USE_DET_FUN,
            fsample=fsample)
        
    elif 0:
        h1 = 32
        num_layers = 3
        
        h1 = 16
        num_layers = 2

        h1 = 16
        num_layers = 3

        h1 = 8
        num_layers = 2
        
        if 1:
            print("model LSTM")
            model = LSTM(input_size=max_sig_num, hidden_dim=h1, output_size=1, num_layers=num_layers, dropout=dropout, useDetFun=USE_DET_FUN, fsample=fsample)
        else:
            print("model DUAL")
            model = DUAL(max_sig_num, 1, n_channels, ksize, dropout, fin_dropout, h1, num_layers, dropout)
    
    else:
        
        print("model SimpSos")
        model = SimpSos(
            input_size=max_sig_num,
            output_size=1,
            fin_dropout=fin_dropout,
            fsample=fsample)   
    
else:
    model_fname = "learnTCN_model_2020_02_04_tacho0256.pt"
    model_fname = "learnTCN_model_2020_02_05_cum2.pt"
    model_fname = "learnTCN_model_2020_02_05_cum3.pt"
    model_fname = "learnTCN_model_2020_02_28_n100_nodet0020_onnx/learnTCN_model.pt"
    print("\n--- loading model!!!")
    print("\t'{}'".format(model_fname))
    
    model = torch.load(open(model_fname, "rb"))

# przykładowy forward pass
if False:
    with torch.no_grad():
        outp = model(train_inputs)
        plt.plot(outp[0, :, 0], 'b-')
        plt.show()

# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html


def my_mse_loss(output, target):
    if False:
        ii = 0
        ii = ii + 1
        plt.plot(output[ii].detach(), 'r-', target[ii], 'b-')
        
    # początek nie brany do obliczenia błedu jako rozgrzewka alg. 
    if True:
        ln = int(len(output[0]))
        idl = 2 if MODE_16s else 4        
        st = int(ln / idl)
        output = output[:, st:ln]
        target = target[:, st:ln]
    
    return ((output - target) ** 2).sum() / output.data.nelement()


def my_rmse_loss(output, target):
    return torch.sqrt(F.mse_loss(output, target))


if PRECOMP_TARG_TACHO:
    all_targets_tacho = []
    
    print("comp targ tacho", flush=True, end="")
    for i in range(len(all_targets)):
        tacho = comp_eq_tacho(all_targets[i], wndlen2=wndlen2, interv=hr_interv, valthresh=valthresh, argmax_trick=False)
        all_targets_tacho = all_targets_tacho + [tacho]
        if i % 10 == 0 :
            print(".", flush=True, end="")
    
    print("ok")
    
    all_targets_tacho = torch.stack(all_targets_tacho)
    test_targets_tacho = all_targets_tacho[itest] 
    train_targets_tacho = all_targets_tacho[itrain]

    
def my_cum_loss(output, target):
    return my_tacho_loss(output, target) + 20 * my_sync_loss(output, target)

def i_my_tacho_loss(output, target): 
    return my_tacho_loss(output=output, target=target, wndlen2=wndlen2, hr_interv=hr_interv, valthresh=valthresh, precomp_targ_tacho=PRECOMP_TARG_TACHO)

def i_my_sync_loss(output, target): 
    return my_sync_loss(output=output, target=target, fsample=fsample, sync_time=sync_time, idl_div=idl_div)


criterion = my_rmse_loss
criterion = F.mse_loss
criterion = my_mse_loss
criterion = i_my_tacho_loss
criterion = my_cum_loss
criterion = i_my_sync_loss


if PRECOMP_TARG_TACHO and criterion != my_tacho_loss:
    raise Exception("-- Conf error: PRECOMP_TARG_TACHO and criterion != my_tacho_loss")

sname = "learnTCN_model"
savname = sname + ".pt"        
savname_dict = sname + "_dict.pt"
savname_onnx = sname + ".onnx"
# optimizer = optim.SGD(model.parameters(), lr=lrate)
#optimizer = optim.Adam(model.parameters(), lr=lrate)

# weight_decay chyba zapobiega NAN-om
optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=1e-5)

minvloss = 1e6

hist_train_loss = []
hist_test_loss = []
hist_test_loss_ep = []

if False:
    print("\n*** set_detect_anomaly(True) !!\n")
    torch.autograd.set_detect_anomaly(True)

for k in range(30000):
    
    model.train()
    # batch = random.sample(list(range(len(itrain))), int(len(itrain) * batch_fract))
     
    batch = random.sample(list(range(len(itrain))), batch_size)
    optimizer.zero_grad()
    output = model(train_inputs[batch]).double()
    
    if not PRECOMP_TARG_TACHO:
        loss = criterion(output.squeeze(2), train_targets[batch])
    else:
        loss = criterion(output.squeeze(2), train_targets_tacho[batch])
        
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    # is nan
    if loss.item() != loss.item():
        print("*** LOSS is NaN !!!, break")
        break        
        
    loss.backward()
    
    hist_train_loss += [loss.item()]
    
    if False and model.tcn.network[0].conv1._parameters["bias"].grad is None:
        raise Exception("gradient missing")
    
    optimizer.step()
    # if k > 0 and k % 10 == 0:
    if k == 0: 
        continue
    
    if k % 5 == 0:
        print(".", end="", flush=True)
                
    if k % 20 == 0 or k == 1:  # or k < 20:
        evmodel = model.eval()
        with torch.no_grad():
            # outp = model(test_input.unsqueeze(0))
            outp = evmodel(test_inputs).double()
            
            if not PRECOMP_TARG_TACHO:
                vloss = criterion(outp.squeeze(2), test_targets)
            else:
                vloss = criterion(outp.squeeze(2), test_targets_tacho)
                
            hist_test_loss += [vloss.item()]
            hist_test_loss_ep += [k]
             
            ifstar = ""
            if vloss < minvloss:
                minvloss = vloss
                if k > 500:
                    torch.save(model, savname)
                    torch.save(model.state_dict(), savname_dict)            
                    if True:
                        dummy_inp = torch.randn(train_inputs[0].unsqueeze(0).size())
                        torch.onnx.export(model, dummy_inp, savname_onnx, opset_version=11)
                    ifstar = "*"
      
            if False:
                import onnx
                onnx_model = onnx.load(savname_onnx)
                print(onnx_model.opset_import)
                import onnxruntime as rt
                
                sess = rt.InferenceSession(savname_onnx)
                                    
            sav_hist_test_df = np.array([hist_test_loss_ep, hist_test_loss ]).transpose()     
            sav_hist_test_hdr = ";".join(["epoch", "test_loss"])
            np.savetxt(savname + "_test_loss.csv", sav_hist_test_df, header=sav_hist_test_hdr, delimiter=";")

            sav_hist_train_df = np.array([np.arange(len(hist_train_loss)), hist_train_loss ]).transpose()     
            sav_hist_train_hdr = ";".join(["epoch", "train_loss"])
            np.savetxt(savname + "_train_loss.csv", sav_hist_train_df, header=sav_hist_train_hdr, delimiter=";")
                
            print("| {:5} | {:.5f} | {:.5f}  {}".format(k, loss.item(), vloss.item(), ifstar))
            
            if False:
                plt.plot(outp[0, :, 0], 'b-')
                plt.draw()
                plt.pause(0.001)
                plt.clf()

            if False and k > 0 and k % 300 == 0:
                print("lrate decay, recreate oprtimizer..")
                lrate /= 4
                optimizer = optim.Adam(model.parameters(), lr=lrate)
  
if True:
    plt.plot(hist_train_loss, label="train")
    plt.plot(hist_test_loss_ep, hist_test_loss, label="test")
    plt.legend()
    # plt.show()

with torch.no_grad():
    plt.clf()
    evmodel = model.eval()
    outp = evmodel(test_inputs).squeeze(2)
    for i in range(len(outp)):
        
        pts = find_maxs(outp[i], wndlen2)
                
        plt.clf()
        plt.plot(outp[i], 'r-', test_targets[i], 'b-')
        plt.plot(pts, outp[i, pts], "ko")
        plt.title(i)        
        plt.draw()
        # plt.show()
        plt.pause(2)

    if False:
        print(allPaths[itest[i]])
        i += 1
        i = 0
        
    # eksport csv do weryf onnx w dotnet
    if False:
        idx = np.arange(0, len(outp[i]))
        savedf = np.append(np.array([idx, outp[i].numpy() ]), test_inputs[i].numpy().transpose(), axis=0).transpose()
        savedf.shape
        hdr = ["idx", "outp", ] + ["inp{}".format(i) for i in range(test_inputs[i].size()[1])]
        hdr = ";".join(hdr)
        np.savetxt("export-in-out.csv", savedf, delimiter=";", header=hdr)  # fmt ,newline, header, footer, comments, encoding)

        
with torch.no_grad():
    plt.clf()
    evmodel = model.eval()
    outp = evmodel(test_inputs).squeeze(2)
    for i in range(len(outp)):
        plt.clf()
        outtacho = comp_eq_tacho(outp[i], wndlen2=wndlen2, interv=hr_interv, valthresh=valthresh, argmax_trick=True)
        outtacho_n = comp_eq_tacho(outp[i], wndlen2=wndlen2, interv=hr_interv, valthresh=valthresh, argmax_trick=False)
        targtacho = comp_eq_tacho(test_targets[i], wndlen2=wndlen2, interv=hr_interv, valthresh=valthresh, argmax_trick=False)
        if PRECOMP_TARG_TACHO:
            plt.plot(test_targets_tacho[i], 'g-')
        plt.plot(outtacho_n, 'k-', outtacho, 'r-', targtacho, 'b-')
        plt.title(i)        
        plt.draw()
        # plt.show()
        plt.pause(2)

if pw != '':
    os.system('echo {} | sudo -kS shutdown +15'.format(pw)) 
if False:
    os.system('echo {} | sudo -kS shutdown -c'.format(pw)) 

print("done")
