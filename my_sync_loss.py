import scipy.signal as signal
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def my_sync_loss(output, target, fsample, sync_time, idl_div):
    if False:
        ii = 0
        ii = ii + 1
        plt.plot(output[ii].detach(), 'r-', target[ii], 'b-')
        
    # początek nie brany do obliczenia błedu jako rozgrzewka alg. 
    ln = int(len(output[0]))
    st = ln // idl_div

    # seloutput = output[:, st:ln]
    # seltarget = target[:, st:ln]

    ln2 = ln // 2
    dt = int(fsample * sync_time)
    
    # output = output[:, st:ln]
    # target = target[:, st:ln]
 
    syncout = []
    synctarg = []
            
    lags = []
    for i in range(len(target)):
        if False:
            corr = signal.correlate(output[i].detach(), target[i].detach(), mode='same')
            lag = np.argmax(corr[ln2 - dt:ln2 + dt])
        else:
            corr = F.conv1d(output[i].unsqueeze(0).unsqueeze(1), target[i].unsqueeze(0).unsqueeze(1), padding=dt).squeeze(0).squeeze(0)
            _, lag = torch.max(corr, 0)
        
        lag = lag - dt
         
        lags = lags + [lag]
          
        if lag < 0:
            out = output[i, st + lag:ln + lag] 
            targ = target[i, st:ln] 
        else:
            out = output[i, st:ln]
            targ = target[i, st - lag:ln - lag] 
    
        syncout.append(out)
        synctarg.append(targ)        
            
    seloutput = torch.stack(syncout)
    seltarget = torch.stack(synctarg)
             
    if False:
        ii = 0
        ii += 1
        nosynout = output[:, st:ln]
        nosynctarg = target[:, st:ln]
        plt.plot(nosynout[ii].detach(), 'gray', nosynctarg[ii], 'black') 
        plt.plot(seloutput[ii].detach(), 'r-', seltarget[ii], '-b')
              
    if False:
        plt.plot(corr.detach())
        plt.plot(corr + 1)
        
    # return ((output - target) ** 2).sum() / output.data.nelement()
    result = ((seloutput - seltarget) ** 2).sum() / seloutput.data.nelement()
    if result != result:
        print("*** LOSS is NaN !!!")
        
        
    return result 
    
    # return (torch.abs(output - target)).sum() / output.data.nelement()
    
    # return (torch.min(torch.abs(output - target), torch.full(output.shape, 0.1, dtype=torch.float64)) / 0.1).sum() / output.data.nelement()
    
    # return (torch.min(((seloutput - seltarget) ** 2), torch.full(seltarget.shape, thresh, dtype=torch.float64)) / thresh).sum() / seltarget.data.nelement()
    # return (torch.max(((seloutput - seltarget) ** 2), torch.full(seloutput.shape, thresh, dtype=torch.float64))).sum() / seloutput.data.nelement()
