import math
import numpy as np
from scipy.fftpack import fft

def DE_PSD(data, fre=200, time_window=0.5, which="de"):
    """
    Compute Differential Entropy (DE) and/or Power Spectral Density (PSD).
    Copied/Adapted from original DE_PSD.py to be importable.
    """
    STFTN = 200
    fStart = [1, 4, 8, 14, 31]
    fEnd = [4, 8, 14, 31, 99]
    window = time_window
    fs = fre

    WindowPoints = fs * window
    fStartNum = np.zeros([len(fStart)], dtype=int)
    fEndNum = np.zeros([len(fEnd)], dtype=int)
    for i in range(0, len(fStart)):
        fStartNum[i] = int(fStart[i] / fs * STFTN)
        fEndNum[i] = int(fEnd[i] / fs * STFTN)

    n = data.shape[0] # Channels
    # m = data.shape[1] # Time points

    if which in ("both", "psd"):
        psd = np.zeros((n, len(fStart)), dtype=float)
    else:
        psd = None

    if which in ("both", "de"):
        de = np.zeros((n, len(fStart)), dtype=float)
    else:
        de = None
        
    Hlength = int(window * fs)
    Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength + 1)) for n in range(1, Hlength + 1)])

    dataNow = data # Assuming data is already the window we want? 
    # Or does DE_PSD expect to slide? Original code: dataNow = data[0:n] implies channels.
    # The loop `for j in range(n)` iterates channels.
    
    for j in range(n):
        temp = dataNow[j]
        if len(temp) != len(Hwindow):
             if len(temp) > len(Hwindow):
                 temp = temp[:len(Hwindow)]
             else:
                 temp = np.pad(temp, (0, len(Hwindow) - len(temp)), 'constant')
        
        Hdata = temp * Hwindow
        FFTdata = fft(Hdata, STFTN)
        magFFTdata = abs(FFTdata[0 : int(STFTN / 2)])
        for p in range(len(fStart)):
            E = 0
            for p0 in range(fStartNum[p] - 1, fEndNum[p]):
                E += magFFTdata[p0] * magFFTdata[p0]
            E = E / (fEndNum[p] - fStartNum[p] + 1)
            if psd is not None:
                psd[j][p] = E
            if de is not None:
                # E can be 0? math.log(0) error?
                if E <= 0:
                    de[j][p] = 0 # Handle log(0)
                else:
                    de[j][p] = math.log(100 * E, 2)
    
    if which == "de":
        return de
    if which == "psd":
        return psd
    return de, psd

def DE_PSD_torch_or_numpy(data):
    # Wrapper to be called from model
    # data: numpy array (Channels, Time)
    return DE_PSD(data, fre=200, time_window=0.5, which="de")
