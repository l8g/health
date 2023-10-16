
import numpy as np
from scipy.sparse import spdiags
from scipy import signal
import torch
import neurokit2 as nk


def detrend(signal, Lambda):

    signal_length = len(signal)
    H = np.identity(signal_length)

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, signal_length - 2, signal_length).toarray()
    filterd_signal = np.dot(H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D)), signal)
    return filterd_signal

def BPF(input_val, fs = 30, low = 0.75, high = 2.5):
    low = low / (fs / 2)
    high = high / (fs / 2)
    b, a = signal.butter(3, [low, high], 'bandpass')
    if type(input_val) == torch.Tensor:
        return signal.filtfilt(b, a, np.double(input_val.cpu().numpy()))
    else:
        return signal.filtfilt(b, a, np.double(input_val))
    
def get_hrv(ppg_signal, fs=30.):
    ppg_peaks = nk.ppg_findpeaks(ppg_signal, sampling_rate=fs)['PPG_Peaks']
    hrv = nk.signal_rate(ppg_peaks, sampling_rate=fs, desired_length=len(ppg_signal))
    return hrv