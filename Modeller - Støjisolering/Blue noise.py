import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch


def estimate_snr_highband(audio_signal, sample_rate, noise_duration_s=0.1):
    noise_samples = int(sample_rate * noise_duration_s)
    if noise_samples >= len(audio_signal):
        noise_samples = len(audio_signal) // 4

    nperseg = min(512, noise_samples)

    freqs, psd_full  = welch(audio_signal, fs=sample_rate, nperseg=nperseg)
    freqs, psd_noise = welch(audio_signal[:noise_samples], fs=sample_rate, nperseg=nperseg)

    high_band = freqs > (sample_rate / 4)
    noise_power  = np.mean(psd_noise[high_band])
    signal_power = np.mean(psd_full[high_band])

    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def select_filter_params(snr_db, sample_rate):
    nyquist = sample_rate / 2
    # Lavere SNR = mere støj = lavere cutoff (mere aggressiv filtrering)
    if snr_db < 5:
        return nyquist * 0.25, 6   # cutoff_hz, filter order
    elif snr_db < 10:
        return nyquist * 0.40, 5
    elif snr_db < 20:
        return nyquist * 0.55, 4
    else:
        return nyquist * 0.70, 3


def apply_blue_noise_filter_iir(audio_signal, sample_rate, cutoff_hz, order):
    nyquist = sample_rate / 2
    cutoff_norm = cutoff_hz / nyquist
    # Klem cutoff væk fra 0 og 1 for at undgå numeriske fejl
    cutoff_norm = np.clip(cutoff_norm, 0.01, 0.99)
    sos = butter(order, cutoff_norm, btype="low", output="sos")
    filtered_signal = sosfiltfilt(sos, audio_signal)
    return filtered_signal


def plot_signals(original, filtered, sample_rate):
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
    plt.title("Filtered signal (IIR Butterworth low-pass)")
    plt.xlabel("Time (s)")
    plt.ylim(y_min, y_max)

    plt.subplot(3, 1, 3)
    plt.plot(time, removed_noise)
    plt.title("Removed noise (Original - Filtered)")
    plt.xlabel("Time (s)")
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


def plot_frequency_response(original, filtered, sample_rate):
    # Sammenlign spektrum før og efter filtrering
    freqs_o, psd_o = welch(original,  fs=sample_rate, nperseg=1024)
    freqs_f, psd_f = welch(filtered, fs=sample_rate, nperseg=1024)

    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs_o, psd_o, label="Original", alpha=0.7)
    plt.semilogy(freqs_f, psd_f, label="Filtered", alpha=0.7)
    plt.title("Power Spectral Density — Original vs Filtered")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.legend()
    plt.tight_layout()
    plt.show()


def remove_blue_noise_from_wav(input_wav_path, output_wav_path):
    # Trin 1: Læs lydfil og gør mono
    audio_signal, sample_rate = sf.read(input_wav_path)
    if audio_signal.ndim > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    # Trin 2: Estimér SNR i høj-frekvensbåndet
    snr_db = estimate_snr_highband(audio_signal, sample_rate)

    # Trin 3: Vælg cutoff og filterorden baseret på SNR
    cutoff_hz, order = select_filter_params(snr_db, sample_rate)
    print(f"Estimeret SNR (høj bånd): {snr_db:.1f} dB")
    print(f"Butterworth low-pass → cutoff: {cutoff_hz:.0f} Hz, orden: {order}")

    # Trin 4: Anvend IIR Butterworth filter (zero-phase via sosfiltfilt)
    filtered_signal = apply_blue_noise_filter_iir(audio_signal, sample_rate, cutoff_hz, order)

    # Trin 5: Gem resultat
    output_folder = os.path.dirname(output_wav_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    sf.write(output_wav_path, filtered_signal, sample_rate)
    print(f"Gemt filtreret fil: {output_wav_path}")

    # Trin 6: Plot tidssignaler og frekvensrespons
    plot_signals(original=audio_signal, filtered=filtered_signal, sample_rate=sample_rate)
    plot_frequency_response(original=audio_signal, filtered=filtered_signal, sample_rate=sample_rate)


input_path  = "C:\\Users\\TOTAL TECH\\Desktop\\Speach_augmented\\Training\\BlueNoise\\BlueNoise_beta-0.90_snr-0.31_193.wav"
output_path = "C:\\Users\\TOTAL TECH\\Desktop\\Speach_augmented\\Training\\BlueNoise\\Uden\\filtered_blue.wav"

remove_blue_noise_from_wav(
    input_wav_path=input_path,
    output_wav_path=output_path,
)