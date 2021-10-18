import numpy as np
from locals import LocalMaxima


def find_maxs(sig, wndlen2):
    ipts = []

    def on_max(idx, value):
        nonlocal ipts
        ipts = ipts + [idx]

    lc1 = LocalMaxima(wndlen2, on_max)
    
    idx = 0
    for samp in sig:
        lc1.next(idx, samp)
        idx = idx + 1
    return ipts     


def comp_hr(sig1d, wndlen2,  valthresh, fsample):
    
    # tachosig = []
    hrsig = np.empty([0, 2])

    pts = find_maxs(sig1d, wndlen2)
    pt = None
    
    for i in range(len(pts)):            
        if sig1d[pts[i]] < valthresh:
            continue
        ppt = pt
        pt = pts[i]
        
        if not ppt is None:
            dt = (pt - ppt) / fsample
            #hr = 60.0 / dt if dt > 60.0 / 300 else 0
            hr = dt
            hrsig = np.append(hrsig, np.array([[pt, hr]]), axis=0)

    return hrsig

