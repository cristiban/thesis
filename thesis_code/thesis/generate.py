from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.signal import spectrogram, resample

def generate_spectrogram(stimulus, response, sr_audio=100):
    scaler = MinMaxScaler((0, 1))
    spec = np.array([spectrogram(trial, fs=sr_audio)[2] for trial in stimulus])
    spec = np.array([scaler.fit_transform(resample(Sxx, response.shape[1], axis=1).T) for Sxx in spec])
    return spec