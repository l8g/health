
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

def detrend_torch(signals, Lambda=100):
    """
    Detrend 1D signals with diagonal matrix D, using torch batch matrix multiplication

    :param signals: Singals with linear trend
    :param Lambda:
    :return:
    """
    test_n, length = signals.shape

    H = torch.eye(length)
    ones = torch.ones(length - 2)

    diag1 = torch.cat((torch.diag(ones), torch.zeros((length - 2, 2))), dim=-1)
    diag2 = torch.cat((torch.zeros((length - 2, 1)), torch.diag(-2 * ones), torch.zeros((length - 2, 1))), dim=-1)
    diag3 = torch.cat((torch.zeros((length - 2, 2)), torch.diag(ones)), dim=-1)
    D = diag1 + diag2 + diag3

    detrended_signal = torch.bmm(signals.unsqueeze(1),
                                 (H - torch.linalg.inv(H + (Lambda ** 2) * torch.t(D) @ D)).to('cuda').expand(test_n,
                                                                                                              -1,
                                                                                                              -1)).squeeze()
    return detrended_signal

def normalize_torch(input_val):
    if type(input_val) != torch.Tensor:
        input_val = torch.from_numpy(input_val.copy())
    min = torch.min(input_val, dim=-1, keepdim=True)[0]
    max = torch.max(input_val, dim=-1, keepdim=True)[0]
    return (input_val - min) / (max - min)

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

def get_hr(rppg, bvp, vital_type = 'HR', cal_type = 'FFT', fs = 30, bpf = None):
    if cal_type == "FFT" and vital_type == "HRV":
        raise ValueError("FFT cannot calculate HRV, To calculate HRV, use 'PEAK' methord instead")
    if cal_type not in ["FFT", "PEAK"]:
        raise ValueError("cal_type must be 'FFT' or 'PEAK'")
    

    bvp = detrend_torch(torch.cumsum(bvp, dim=-1))
    rppg = detrend_torch(torch.cumsum(rppg, dim=-1))
    if bpf != 'None':
        low, high = bpf
        bvp = normalize_torch(BPF(bvp, fs, low, high))
        rppg = normalize_torch(BPF(rppg, fs, low, high))
    else:
        bvp = normalize_torch(bvp)
        rppg = normalize_torch(rppg)
    
    hr_pred = calc_hr_torch(cal_type, rppg, fs)
    hr_target = calc_hr_torch(cal_type, bvp, fs)

    if cal_type == 'PEAK':
        hr_pred, hrv_pred, index_pred = hr_pred
        hr_target, hrv_target, index_target = hr_target

        if vital_type == 'HRV':
            return hrv_pred, hrv_target
    
    return hr_pred, hr_target




def calc_hr_torch(calc_type, ppg_signals, fs = 30.):
    test_n, sig_length = ppg_signals.shape
    hr_list = torch.empty(test_n)
    if calc_type == 'FFT':
        ppg_signals = ppg_signals - torch.mean(ppg_signals, dim=-1, keepdim=True)
        N = sig_length
        k = torch.arange(N)
        T = N / fs
        freq = k / T
        amplitude = torch.abs(torch.fft.rfft(ppg_signals, n=N, dim=-1)) / N
        hr_list = freq[torch.argmax(amplitude, dim=-1)] * 60
        return hr_list
    elif calc_type == 'PEAK':
        hrv_list = -torch.ones((test_n, sig_length // fs * 10))
        index_list = -torch.ones((test_n, sig_length // fs * 10))
        width = 11
        window_maxina = torch.nn.functional.max_pool1d(ppg_signals, width, 1, padding=width // 2, return_indices=True)[1].squeeze()
        for i in range(test_n):
            candidate = window_maxina[i].unique()
            nice_peaks = candidate[window_maxina[i][candidate] == candidate]
            nice_peaks = nice_peaks[ppg_signals[i][nice_peaks] > torch.mean(ppg_signals[i][nice_peaks] / 2)]
            beat_interval = torch.diff(nice_peaks)
            hrv = beat_interval / fs
            hr = torch.mean(60 / hrv)
            hr_list[i] = hr
            hrv_list[i, :len(hrv)] = hrv * 1000
            index_list[i, :len(nice_peaks)] = nice_peaks
        
        hrv_list = hrv_list[:, :torch.max(torch.sum(hrv_list > 0, dim=-1))]
        index_list = index_list[:, :torch.max(torch.sum(index_list > 0, dim=-1))]

        return hr_list, hrv_list, index_list
    
def MAE(pred, label):
    return np.mean(np.abs(pred - label))

def RMSE(pred, label):
    return np.sqrt(np.mean(np.square(pred - label)))

def MAPE(pred, label):
    return np.mean(np.abs((pred - label) / label)) * 100

def corr(pred, label):
    return np.corrcoef(pred, label)

def SD(pred, label):
    return np.std(pred - label)