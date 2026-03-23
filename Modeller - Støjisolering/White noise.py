import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


#Samlede process:
# 1. Læs lydfil og gør den mono
# 2. Estimér støjniveau (SNR) fra de første 100ms af lydklippet
# 3. Vælg filterstyrke baseret på SNR (window lenght)
# 4. Anvend moving average FIR-filter
# 5. Gem filtreret lyd
# 6. Plot original, filtreret og fjernet støj


def estimate_snr(audio_signal, sample_rate, noise_duration_s=0.1):
    # Beregn RMS af de første 100ms (antaget støj) og hele signalet
    # Returnér forholdet mellem dem i dB
    noise_samples = int(sample_rate * noise_duration_s)
    noise_rms = np.sqrt(np.mean(audio_signal[:noise_samples] ** 2))
    signal_rms = np.sqrt(np.mean(audio_signal ** 2))
    if noise_rms == 0:
        return float("inf")
    return 20 * np.log10(signal_rms / noise_rms)


def select_window_length(snr_db):
    # Lav støj (høj SNR) → lille vindue (mild filtrering)
    # Høj støj (lav SNR) → stort vindue (aggressiv filtrering)
    if snr_db < 5:
        return 21
    elif snr_db < 10:
        return 15
    elif snr_db < 20:
        return 9
    else:
        return 5


def apply_moving_average_fir_filter(audio_signal, window_length):
    # Glat signalet ved at tage gennemsnittet af naboværdier
    fir_coefficients = np.ones(window_length) / window_length
    filtered_signal = np.convolve(audio_signal, fir_coefficients, mode="same")
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


def remove_white_noise_from_wav(input_wav_path, output_wav_path):
    # Trin 1: Læs lydfil og gør mono
    audio_signal, sample_rate = sf.read(input_wav_path)
    if audio_signal.ndim > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    # Trin 2-3: Estimér SNR og vælg filterstyrke automatisk
    snr_db = estimate_snr(audio_signal, sample_rate)
    window_length = select_window_length(snr_db)
    print(f"Estimeret SNR: {snr_db:.1f} dB  →  window_length: {window_length}")

    # Trin 4: Filtrér signalet
    filtered_signal = apply_moving_average_fir_filter(audio_signal, window_length)

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


input_path  = "/Users/jonassvirkaer/Desktop/Speach_augmented/Training/WhiteNoise/WhiteNoise_beta0.10_snr-4.38_174.wav"
output_path = "/Users/jonassvirkaer/Desktop/Uden/filtered.wav"

remove_white_noise_from_wav(
    input_wav_path=input_path,
    output_wav_path=output_path,
)