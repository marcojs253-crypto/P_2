import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

# Samlede process:
# 1. Læs lydfil og gør den mono
# 2. Estimér dominerende støjfrekvens via PSD af de første 100ms
# 3. Estimér SNR fra de første 100ms og vælg filterstyrke (cutoff + orden)
# 4. Anvend Butterworth high-pass filter
# 5. Gem filtreret lyd
# 6. Plot original, filtreret og fjernet støj

def find_dominant_noise_freq(noise_segment, sample_rate):
    # Beregn power spectral density (PSD) for støjsegmentet
    # Kig kun under 500 Hz, da brown noise dominerer de lave frekvenser
    # Returnér frekvensen med højest effekt
    freqs, psd = welch(noise_segment, sample_rate, nperseg=1024)
    low_freq_mask = freqs < 500
    dominant_freq = freqs[low_freq_mask][np.argmax(psd[low_freq_mask])]
    return dominant_freq

def estimate_snr(audio_signal, sample_rate, noise_duration_s=0.1):
    # Beregn RMS af de første 100ms (antaget støj) og hele signalet
    # Returnér forholdet mellem dem i dB
    noise_samples = int(sample_rate * noise_duration_s)
    noise_rms = np.sqrt(np.mean(audio_signal[:noise_samples] ** 2))
    signal_rms = np.sqrt(np.mean(audio_signal ** 2))
    if noise_rms == 0:
        return float("inf")
    return 20 * np.log10(signal_rms / noise_rms)

def select_filter_params(snr_db, dominant_freq):
    # Lav støj (høj SNR) → cutoff tæt på dominant_freq, lav orden (mild filtrering)
    # Høj støj (lav SNR) → cutoff højere end dominant_freq, høj orden (aggressiv filtrering)
    # Grænser er skubbet op så filteret hurtigere falder i den milde kategori
    if snr_db < 10:
        return dominant_freq * 1.5, 5
    elif snr_db < 25:
        return dominant_freq * 1.2, 3
    else:
        return dominant_freq * 1.1, 2

def apply_highpass_filter(audio_signal, sample_rate, cutoff_hz, order):
    # Design et Butterworth high-pass filter med given cutoff og orden
    # filtfilt sikrer nul faseforskydning (kører filteret frem og tilbage)
    nyquist = sample_rate / 2
    b, a = butter(order, cutoff_hz / nyquist, btype='high', analog=False)
    filtered_signal = filtfilt(b, a, audio_signal)
    return filtered_signal

def plot_signals(original, filtered, sample_rate):
    # Vis original, filtreret og fjernet støj med samme y-akse
    time = np.arange(len(original)) / sample_rate
    removed_noise = original - filtered
    y_min = min(np.min(original), np.min(filtered), np.min(removed_noise))
    y_max = max(np.max(original), np.max(filtered), np.max(removed_noise))

    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, original)
    plt.title("Original signal")
    plt.xlabel("Time (s)")
    plt.ylim(y_min, y_max)

    plt.subplot(3, 1, 2)
    plt.plot(time, filtered)
    plt.title("Filtered signal")
    plt.xlabel("Time (s)")
    plt.ylim(y_min, y_max)

    plt.subplot(3, 1, 3)
    plt.plot(time, removed_noise)
    plt.title("Removed noise (Original - Filtered)")
    plt.xlabel("Time (s)")
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()

def remove_brown_noise_from_wav(input_wav_path, output_wav_path):
    # Trin 1: Læs lydfil og gør mono
    audio_signal, sample_rate = sf.read(input_wav_path)
    if audio_signal.ndim > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    # Trin 2: Find dominerende støjfrekvens i de første 100ms
    noise_samples = int(sample_rate * 0.1)
    dominant_freq = find_dominant_noise_freq(audio_signal[:noise_samples], sample_rate)

    # Trin 3: Estimér SNR og vælg filterstyrke automatisk
    snr_db = estimate_snr(audio_signal, sample_rate)
    cutoff_hz, order = select_filter_params(snr_db, dominant_freq)
    print(f"Dominerende støjfrekvens: {dominant_freq:.1f} Hz")
    print(f"Estimeret SNR: {snr_db:.1f} dB  →  cutoff: {cutoff_hz:.1f} Hz, orden: {order}")

    # Trin 4: Filtrér signalet
    filtered_signal = apply_highpass_filter(audio_signal, sample_rate, cutoff_hz, order)

    # Trin 5: Gem resultat
    output_folder = os.path.dirname(output_wav_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    sf.write(output_wav_path, filtered_signal, sample_rate)

    # Trin 6: Plot signalerne
    plot_signals(
        original=audio_signal,
        filtered=filtered_signal,
        sample_rate=sample_rate,
    )

input_path  = "/Users/jonassvirkaer/Desktop/Speach_augmented/Training/BrownNoise/BrownNoise_beta1.90_snr-3.36_168.wav"
output_path = "/Users/jonassvirkaer/Desktop/Uden/filtered.wav"
remove_brown_noise_from_wav(
    input_wav_path=input_path,
    output_wav_path=output_path,
)
