import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert, butter, filtfilt
from scipy.io.wavfile import write

# === Step 1: Load modulated audio ===
sample_rate, data = wavfile.read("modulated_noisy_audio.wav")

# If stereo, take one channel
if data.ndim > 1:
    data = data[:, 0]

# Normalize signal
data = data / np.max(np.abs(data))

# Time vector
time = np.linspace(0, len(data) / sample_rate, len(data))

# === Step 2: FFT to estimate carrier frequency ===
n = len(data)
freq = np.fft.fftfreq(n, d=1/sample_rate)
fft_spectrum = np.abs(np.fft.fft(data))

# Keep only positive frequencies
positive_freqs = freq[:n // 2]
positive_spectrum = fft_spectrum[:n // 2]

# Find peak (carrier frequency)
carrier_freq = positive_freqs[np.argmax(positive_spectrum)]
print(f"Estimated Carrier Frequency: {carrier_freq:.2f} Hz")

# === Step 3: Demodulate using Hilbert Transform ===
analytic_signal = hilbert(data)
envelope = np.abs(analytic_signal)

# === Step 4: Low-pass filter the envelope ===
def butter_lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    norm_cutoff = cutoff / nyquist
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

cutoff_freq = 5000  # 5kHz is typical for audio
demodulated = butter_lowpass_filter(envelope, cutoff=cutoff_freq, fs=sample_rate)

# Normalize output for saving
demodulated_normalized = demodulated / np.max(np.abs(demodulated))
write("demodulated_output.wav", sample_rate, (demodulated_normalized * 32767).astype(np.int16))

# === Step 5: Plot Signals ===
plt.figure(figsize=(15, 10))

# Plot original (modulated) signal
plt.subplot(3, 1, 1)
plt.plot(time[:2000], data[:2000])
plt.title("Modulated Signal (Time Domain)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# Plot frequency spectrum
plt.subplot(3, 1, 2)
plt.plot(positive_freqs, positive_spectrum)
plt.axvline(carrier_freq, color='r', linestyle='--', label=f"Carrier ≈ {carrier_freq:.1f} Hz")
plt.title("FFT Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.legend()

# Plot demodulated signal
plt.subplot(3, 1, 3)
plt.plot(time[:2000], demodulated[:2000], color='g')
plt.title("Demodulated Signal (Envelope)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show() 