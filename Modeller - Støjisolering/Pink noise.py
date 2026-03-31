import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch

import os
for f in os.listdir("C:\\Audio\\Speach_augmented\\Training\\PinkNoise"):
    print(f)
def estimate_snr_lowband(audio_signal, sample_rate, noise_duration_s=0.1):
    """
    Estimate SNR in the LOW frequency band (below sample_rate / 4).
    Pink noise energy is concentrated at low frequencies (-3 dB/octave),
    so this is the relevant band to measure noise vs signal power.
    """
    noise_samples = int(sample_rate * noise_duration_s)
    if noise_samples >= len(audio_signal):
        noise_samples = len(audio_signal) // 4

    nperseg = min(512, noise_samples)

    freqs, psd_full  = welch(audio_signal,                 fs=sample_rate, nperseg=nperseg)
    freqs, psd_noise = welch(audio_signal[:noise_samples], fs=sample_rate, nperseg=nperseg)

    low_band = freqs < (sample_rate / 4)

    noise_power  = np.mean(psd_noise[low_band])
    signal_power = np.mean(psd_full[low_band])

    if noise_power <= 0:
        return float("inf")

    snr_db = 10 * np.log10(max(signal_power, 1e-12) / noise_power)
    return snr_db


def select_filter_params(snr_db, sample_rate):
    nyquist = sample_rate / 2

    if snr_db < 5:
        return 200, 2    # fixed Hz, not a Nyquist fraction
    elif snr_db < 10:
        return 150, 2
    elif snr_db < 20:
        return 100, 1
    else:
        return 80,  1


def apply_pink_noise_filter_iir(audio_signal, sample_rate, cutoff_hz, order):
    """
    Apply a zero-phase Butterworth HIGH-pass filter.
    High-pass removes the low-frequency energy where pink noise dominates.
    sosfiltfilt ensures zero phase distortion (no time-shift artifacts).
    """
    nyquist = sample_rate / 2
    cutoff_norm = cutoff_hz / nyquist
    # Clamp away from 0 and 1 to avoid numerical instability in butter()
    cutoff_norm = np.clip(cutoff_norm, 0.01, 0.99)
    sos = butter(order, cutoff_norm, btype="high", output="sos")
    filtered_signal = sosfiltfilt(sos, audio_signal)
    return filtered_signal


def plot_signals(original, filtered, sample_rate):
    time = np.arange(len(original)) / sample_rate
    removed_noise = original - filtered

    y_min = min(np.min(original), np.min(filtered), np.min(removed_noise))
    y_max = max(np.max(original), np.max(filtered), np.max(removed_noise))

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, original, color="steelblue")
    plt.title("Original signal")
    plt.xlabel("Time (s)")
    plt.ylim(y_min, y_max)

    plt.subplot(3, 1, 2)
    plt.plot(time, filtered, color="seagreen")
    plt.title("Filtered signal (IIR Butterworth high-pass)")
    plt.xlabel("Time (s)")
    plt.ylim(y_min, y_max)

    plt.subplot(3, 1, 3)
    plt.plot(time, removed_noise, color="tomato")
    plt.title("Removed noise (Original - Filtered)")
    plt.xlabel("Time (s)")
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


def plot_frequency_response(original, filtered, sample_rate):
    """Compare power spectral density before and after filtering."""
    freqs_o, psd_o = welch(original, fs=sample_rate, nperseg=1024)
    freqs_f, psd_f = welch(filtered, fs=sample_rate, nperseg=1024)

    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs_o, psd_o, label="Original", alpha=0.7, color="steelblue")
    plt.semilogy(freqs_f, psd_f, label="Filtered", alpha=0.7, color="seagreen")
    plt.title("Power Spectral Density — Original vs Filtered")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.legend()
    plt.tight_layout()
    plt.show()


def remove_pink_noise_from_wav(input_wav_path, output_wav_path, noise_duration_s=0.1):
    # Step 1: Read audio file and convert to mono
    audio_signal, sample_rate = sf.read(input_wav_path)
    audio_signal, sample_rate = sf.read(input_wav_path)
    print(f"Sample rate: {sample_rate} Hz, Shape: {audio_signal.shape}, Duration: {len(audio_signal)/sample_rate:.2f}s")
    if audio_signal.ndim > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    # Step 2: Estimate SNR in the low-frequency band (where pink noise dominates)
    snr_db = estimate_snr_lowband(audio_signal, sample_rate, noise_duration_s)

    # Step 3: Handle edge cases before selecting filter params
    if snr_db == float("inf"):
        print("SNR is infinite (no noise detected). Saving original file unchanged.")
        sf.write(output_wav_path, audio_signal, sample_rate)
        return

    if snr_db <= 0:
        print(f"Warning: SNR is {snr_db:.1f} dB — signal is weaker than noise in low band.")
        print("Applying most aggressive filter settings.")
        snr_db = -1  # force lowest branch in select_filter_params

    # Step 4: Choose cutoff and filter order based on SNR
    cutoff_hz, order = select_filter_params(snr_db, sample_rate)
    print(f"Estimated SNR (low band): {snr_db:.1f} dB")
    print(f"Butterworth high-pass → cutoff: {cutoff_hz:.0f} Hz, order: {order}")

    # Step 5: Apply IIR Butterworth high-pass filter (zero-phase)
    filtered_signal = apply_pink_noise_filter_iir(audio_signal, sample_rate, cutoff_hz, order)

    # Step 6: Clip output to safe range before writing
    filtered_signal = np.clip(filtered_signal, -1.0, 1.0)

    # Step 7: Save result
    output_folder = os.path.dirname(output_wav_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    sf.write(output_wav_path, filtered_signal, sample_rate)
    print(f"Saved filtered file: {output_wav_path}")

    # Step 8: Plot time-domain signals and frequency response
    plot_signals(original=audio_signal, filtered=filtered_signal, sample_rate=sample_rate)
    plot_frequency_response(original=audio_signal, filtered=filtered_signal, sample_rate=sample_rate)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    input_path  = "C:\\Audio\\Speach_augmented\\Training\\PinkNoise\\PinkNoise_beta0.90_snr12.04_292.wav"
    output_path = "C:\\Audio\\Try\\PinkNoise_beta0.90_snr12.04_292.wav"

    remove_pink_noise_from_wav(
        input_wav_path=input_path,
        output_wav_path=output_path,
        noise_duration_s=0.1,   # first 100 ms assumed to be noise
    )