import numpy as np
import scipy.signal as signal
import pyreadr
import matplotlib.pyplot as plt
import math

def my_resample(x, q):
    b, a = signal.butter(5, .8 / q)
    zi = x[0] * signal.lfilter_zi(b, a) 
    y = signal.lfilter(b, a, x, zi=zi)[0][slice(0, len(x), q)]
    return y


def my_resample_m_obso(x, q):
    'downsample wg osi 1, oś 0 to kolejne kanały'
    b, a = signal.butter(5, .8 / q)
    zi0 = signal.lfilter_zi(b, a)
    ziA = zi0.reshape(1, len(zi0))
    ziB = x[:, 0].reshape(x.shape[0], 1)
    zi = np.dot(ziB, ziA)
    y, _ = signal.lfilter(b, a, x, zi=zi, axis=1)
    return y[:, slice(0, x.shape[1], q)]

def my_resample_m(x, q):
    'downsample wg osi 1, oś 0 to kolejne kanały'
    sos = signal.butter(5, .8 / q, output='sos')
    zi0 = signal.sosfilt_zi(sos)
    
    nsig = len(x)
    nsec = len(sos)
    zi = np.repeat(zi0[:, None, :], nsig, 1)
    
    scal = x[:, 0]
    scal = np.repeat(scal[None, :], nsec, 0)
    scal = np.repeat(scal[:, :, None], 2, 2)
    
    # zi.shape == (nsec, nsig, 2)
    y, _ = signal.sosfilt(sos, x, zi=zi * scal, axis=1)
    return y[:, slice(0, x.shape[1], q)]


def filter_hr(sig, fsample, bp1, bp2):
    b, a = signal.butter(2, [bp1 / (fsample / 2), bp2 / (fsample / 2)], "pass")
    zi = sig[0] * signal.lfilter_zi(b, a) 
    return signal.lfilter(b, a, sig, zi=zi)[0]

def filter_hr_m_obso(x, fsample, bp1, bp2):
    'filter_hr wg osi 1, oś 0 to kolejne kanały'
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    
    b, a = signal.butter(2, [bp1 / (fsample / 2), bp2 / (fsample / 2)], "bandpass")
    zi0 = signal.lfilter_zi(b, a)
    ziA = zi0.reshape(1, len(zi0))
    ziB = x[:, 0].reshape(x.shape[0], 1)
    zi = np.dot(ziB, ziA)
    y, _ = signal.lfilter(b, a, x, zi=zi, axis=1)
    return y

def filter_hr_m(x, fsample, bp1, bp2):
    'filter_hr wg osi 1, oś 0 to kolejne kanały'
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    
    sos = signal.butter(2, [bp1 / (fsample / 2), bp2 / (fsample / 2)], "bandpass", output='sos')
    zi0 = signal.sosfilt_zi(sos)
    
    nsig = len(x)
    nsec = len(sos)
    zi = np.repeat(zi0[:, None, :], nsig, 1)
    
    scal = x[:, 0]
    scal = np.repeat(scal[None, :], nsec, 0)
    scal = np.repeat(scal[:, :, None], 2, 2)
  
    # zi.shape == (nsec, nsig, 2)
    y, _ = signal.sosfilt(sos, x, zi=zi * scal, axis=1)

    return y


def load_data(allPaths, fsample, down, siglen, filter=None):
    
    tol = 1e-30
    QUANT_NORM = False
    QUANT_NORM = True       
    
    rsiglen = int(siglen / down)
    all_inputs = np.empty([0, rsiglen , 9])
    all_targets = np.empty([0, rsiglen])
    
    for dataPath, targPath in allPaths:
        
        # print(dataPath)
        result = pyreadr.read_r(dataPath) 
        
        targ = pyreadr.read_r(targPath)
        targ = targ["tsig"].to_numpy()
        
        dim = result["dim"].to_numpy()
        dim = dim.reshape(2)
        
        # chyba pyreadr po upgrade zamienił osie w ładowanych danych !!!
        if False:
            inp_sigs = result["allSig"].to_numpy().reshape(list(reversed(dim)))
        else:
            inp_sigs = result["allSig"].to_numpy().reshape(dim)
            inp_sigs = np.swapaxes(inp_sigs, 1, 0)
    
        # wstępna filtracja HR !!!
        if not filter is None:
            inp_sigs = filter_hr_m(inp_sigs, fsample, filter[0], filter[1])
            
        
        if False:
            bp1 = 6
            bp2 = 36 
            sigHr0 = filter_hr(inp_sigs[1, ], fsample, bp1, bp2)
            plt.plot(sigHr0, 'b-')
            plt.plot(inp_sigs[1, ], 'b-')
        
        
        # sig0dn = signal.resample_poly(sig, 1, 8)

        inp_prep = my_resample_m(inp_sigs, down)
        
        
        if False:
            inp_prep = inp_prep - np.mean(inp_prep, axis=1, keepdims=True)
            #inp_prep = inp_prep / np.std(inp_prep, axis=1, keepdims=True)
            stds = np.std(inp_prep, axis=1, keepdims=True)
            stds = np.where(np.abs(stds) > tol, stds, 1)
            inp_prep = inp_prep / stds
            
        if True:   
            # normowanie online z uzupełnionym startem  
            
            if not QUANT_NORM:
                # liczba odchyleń std do warunku na outliers 
                outstd = 1
             
            # czas pływającego normowania
            normtm = 3.0
            normlen = int(normtm * fsample / down)
            normMaxOutAmp = 100

            pren = np.copy(inp_prep[:, 0:normlen])
            
            if not QUANT_NORM:
                stdp = np.std(pren, axis=1, keepdims=True)
                stdp = np.where(stdp > tol, stdp, 1)
                pren -= np.mean(pren, axis=1, keepdims=True)
                pren /= stdp
                if False:
                    stdpcor = np.zeros(stdp.shape) 
                    for k in range(len(pren)):
                        stdpcor[k, 0] = np.std(pren[k, np.abs(pren[k]) < outstd * stdp[k]])
                    stdpcor = np.where(stdpcor > tol, stdpcor, 1)
                    pren /= stdpcor
            else:
                ipc = np.percentile(pren, [25, 50, 75], axis=1)
                iamp = np.where(ipc[2] - ipc[0] > tol, ipc[2] - ipc[0], 1)
                # pren -= np.mean(pren, axis=1, keepdims=True)
                pren -= ipc[1][:, None]
                pren /= iamp[:, None]                  
                pren = np.where(pren < normMaxOutAmp, pren, normMaxOutAmp)
                pren = np.where(pren > -normMaxOutAmp, pren, -normMaxOutAmp)
                    
            ninp = np.append(pren, np.zeros([inp_prep.shape[0], inp_prep.shape[1] - normlen]), axis=1)
            
            iamp = None
            
            # i = normlen
            for i in range(normlen, len(inp_prep[1])):
                iinp = inp_prep[:, slice(i - normlen, i)]
                # nmn = np.mean(iinp, axis=1)
                # iinp.shape
      
                if QUANT_NORM:
                    if iamp is None or i & 0xF == 0:
                        ipc = np.percentile(iinp, [25, 50, 75], axis=1)
                        iamp = np.where(ipc[2] - ipc[0] > tol, ipc[2] - ipc[0], 1)
                    # ninp[:, i] = (inp_prep[:, i] - nmn) / iamp      
                    ninp[:, i] = (inp_prep[:, i] - ipc[1]) / iamp
                    ninp[:, i] = np.where(ninp[:, i] < normMaxOutAmp, ninp[:, i], normMaxOutAmp)
                    ninp[:, i] = np.where(ninp[:, i] > -normMaxOutAmp, ninp[:, i], -normMaxOutAmp)
                                    
                else:
                    nmn = np.mean(iinp, axis=1)
                    nst = np.std(iinp, axis=1)
                    nst = np.where(np.abs(nst) > tol, nst, 1)
                    ninp[:, i] = (inp_prep[:, i] - nmn) / nst
                    
                    # obso
                    if False:
                        nstcor = np.zeros(len(nst))
                        for k in range(len(iinp)):
                            nstcor[k] = np.std(iinp[k, np.abs(iinp[k]) < outstd * nst[k]])
                        nstcor = np.where(nstcor > tol, nstcor, 1)
                        ninp[:, i] = (inp_prep[:, i] - nmn) / nstcor   
                
            inp_prep = ninp
                    
        if False:
            print("mod scal")
            inp_prep = inp_prep - np.mean(inp_prep)
            inp_prep = inp_prep / np.std(inp_prep)
                
        targ = my_resample_m(np.transpose(targ), down)
        targ = targ[0].astype(np.float64)
    
        if False:
            plt.plot(targ, 'b-')
            bp1 = 6
            bp2 = 36
            inp_prep.shape
            ii = 0
            ii += 1
            plt.plot(ninp[ii, ], 'b-')
            plt.plot(inp_prep[ii, ], 'r-')
            sigHr0 = filter_hr(inp_prep[ii], fsample, bp1, bp2)
            plt.plot(sigHr0, 'b-')
            plt.plot(np.arange(len(sigHr0)) / fsample, sigHr0, 'b-')
        
        #inp = torch.Tensor(np.transpose(inp_prep).astype(np.float64))
        
        # gdy mniej niż 9 siatek uzupełnia zerami
        if inp_prep.shape[0] < 9 :
            addzs = np.zeros([9 - inp_prep.shape[0], inp_prep.shape[1]])
            inp_prep = np.append(inp_prep, addzs, axis=0)
            #print("zero signals added")
            
            
        inp  = np.transpose(inp_prep).astype(np.float64)
        
        #inp = inp.reshape([1, *inp.shape])
        all_inputs = np.append(all_inputs, [inp], axis=0)
        all_targets = np.append(all_targets, [targ], axis=0) 
    
    return all_inputs, all_targets




