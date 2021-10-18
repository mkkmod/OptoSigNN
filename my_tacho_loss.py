import sys
import torch

sys.path.append("../locals")
from locals import LocalMaxima

import matplotlib.pyplot as plt



def find_maxs(sig, wndlen2):
    # ipts = np.empty([0, 2])
    ipts = []

    def on_max(idx, value):
        nonlocal ipts
        # ipts = np.append(ipts, np.array([[idx, value]]), axis=0)
        ipts = ipts + [idx]

    lc1 = LocalMaxima(wndlen2, on_max)
    
    # idx = torch.tensor([0.0], requires_grad=True)
    idx = 0
    for samp in sig:
        # lc1.next(idx.clone(), samp)
        lc1.next(idx, samp)
        idx = idx + 1
    return ipts     


def comp_eq_tacho(sig1d, wndlen2, interv, valthresh, argmax_trick=False):
    
    # tachosig = []
    tachosig = torch.tensor([], dtype=float)
    opts = find_maxs(sig1d, wndlen2)
    oidx = 0
    pt = None
    ppt = None
    for tm in range(0, len(sig1d), interv):
        # while oidx < len(opts) and tm >= opts[oidx, 0]:
        while oidx < len(opts) and tm >= opts[oidx]:
            oidx += 1
            if oidx >= len(opts):
                pt = None
                ppt = None
            # elif opts[oidx, 1] > valthresh:
            elif sig1d[opts[oidx]] > valthresh:
                ppt = pt
                # pt = opts[oidx, 0]
                pt = opts[oidx]
                if argmax_trick:
                    # ipt = int(opts[oidx, 0])
                    ipt = int(opts[oidx])
                    ta = sig1d[ipt - wndlen2 // 2:ipt + wndlen2 // 2] 
                    tb = torch.arange(ipt - wndlen2 // 2, ipt + wndlen2 // 2, dtype=float)
                    pt = torch.sum(ta * tb) / sum(ta)
                else:
                    pt = torch.tensor(pt, dtype=float)
                    
        dt = pt - ppt if not pt is None and not ppt is None else torch.tensor(0.0, dtype=float)
        # tachosig = tachosig + [dt]
        tachosig = torch.cat([tachosig, torch.stack([dt])])
    # return torch.tensor(tachosig)
    return tachosig


def my_tacho_loss(output, target, wndlen2, hr_interv, valthresh, precomp_targ_tacho):

    res = 0

    for i in range(len(output)):
        otach = comp_eq_tacho(output[i], wndlen2, hr_interv, valthresh, argmax_trick=True)
        
        if not precomp_targ_tacho:
            ttach = comp_eq_tacho(target[i], wndlen2, hr_interv, valthresh)
        else:
            ttach = target[i]
 
        res += torch.sqrt(((otach - ttach) ** 2).sum() / len(otach))
        
        if False:
            plt.plot(otach.detach())
            plt.plot(ttach.detach())
            
    return res / len(output[0])
