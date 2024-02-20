import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, decimate
import os
os.chdir(os.path.dirname(__file__))
def load_ppg_data(file_path: str) -> np.ndarray:
    ppg_data = pd.read_csv(file_path)
    return ppg_data['PPG'].values

def preprocess_signal(signal: np.ndarray, fs: int, target_fs: int = 125, bandpass_range: tuple = (0.5, 6)) -> tuple:
    # Band-pass filter design parameters
    lowcut, highcut = bandpass_range
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    
    # Downsample if necessary
    if fs > target_fs:
        decimation_factor = fs // target_fs
        processed_signal = decimate(filtered_signal, decimation_factor, ftype='fir', zero_phase=True)
        processed_fs = target_fs
    else:
        processed_signal = filtered_signal
        processed_fs = fs
    
    return processed_signal, processed_fs


# Load the PPG data
file_path = r"PPG.csv"   # Replace with your file path
fs = 2000  # Assuming the original sampling rate is 2000Hz

ppg_signal = load_ppg_data(file_path)
filtered_signal, current_fs = preprocess_signal(ppg_signal, fs)


# Compute the FFT of the filtered signal to generate noise reference after preprocessing and downsampling
fft_filtered = np.fft.fft(filtered_signal)
frequencies_filtered = np.fft.fftfreq(len(filtered_signal), 1/current_fs)

# Zero out the cardiac and respiratory frequency components as before
fft_filtered[(frequencies_filtered > 0.2) & (frequencies_filtered < 0.35)] = 0
fft_filtered[(frequencies_filtered > 0.5) & (frequencies_filtered < 4)] = 0

# Apply inverse FFT to get the synthetic noise reference signal from the fully processed data
synthetic_noise_final = np.fft.ifft(fft_filtered)


plt.figure(figsize=(18, 12))

# Downsampled PPG Signal
plt.subplot(4, 1, 1)
plt.plot(filtered_signal)
plt.title('filtered PPG Signal (to 125Hz)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

# Final Synthetic Noise Reference Signal
plt.subplot(4, 1, 3)
plt.plot(synthetic_noise_final.real)  # Taking the real part as the output of IFFT is complex
plt.title('Synthetic Noise Reference Signal (Fully Processed)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()