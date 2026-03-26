import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import cheby2, sosfiltfilt, welch


def estimate_snr_highband(audio_signal, sample_rate, noise_duration_s=0.1):
    noise_samples = int(sample_rate * noise_duration_s)
    noise_samples = max(noise_samples, 512)
    nperseg = min(512, noise_samples)

    freqs, psd_noise = welch(audio_signal[:noise_samples], fs=sample_rate, nperseg=nperseg)

    mid_start = len(audio_signal) // 3
    mid_end = min(mid_start + noise_samples * 3, len(audio_signal))
    freqs, psd_mid = welch(audio_signal[mid_start:mid_end], fs=sample_rate, nperseg=nperseg)

    high_band = freqs > (sample_rate / 4)
    noise_power  = np.mean(psd_noise[high_band])
    signal_power = np.mean(psd_mid[high_band])

    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def select_filter_params(snr_db, sample_rate):
    nyquist = sample_rate / 2
    if snr_db < 2:
        return nyquist * 0.15, 8
    elif snr_db < 5:
        return nyquist * 0.20, 7
    elif snr_db < 10:
        return nyquist * 0.35, 6
    elif snr_db < 20:
        return nyquist * 0.50, 5
    else:
        return nyquist * 0.65, 4


def apply_deemphasis(audio_signal, strength=0.95):
    b = np.array([1.0, -strength])
    a = np.array([1.0, -strength * 0.5])
    sos = np.array([[b[0], b[1], 0.0, a[0], a[1], 0.0]])
    return sosfiltfilt(sos, audio_signal)


def apply_blue_noise_filter_iir(audio_signal, sample_rate, cutoff_hz, order, rs=40):
    nyquist = sample_rate / 2
    cutoff_norm = np.clip(cutoff_hz / nyquist, 0.01, 0.99)
    sos = cheby2(order, rs, cutoff_norm, btype="low", output="sos")
    return sosfiltfilt(sos, audio_signal)


def apply_gentle_lowpass(audio_signal, sample_rate, cutoff_hz=8000, order=3):
    # Let post-filter — tager kanten af støj der blev løftet op af forstærkning
    nyquist = sample_rate / 2
    cutoff_norm = np.clip(cutoff_hz / nyquist, 0.01, 0.99)
    sos = cheby2(order, 30, cutoff_norm, btype="low", output="sos")
    return sosfiltfilt(sos, audio_signal)


def soft_limit(audio_signal, threshold=0.95):
    # Blødt loft via tanh — runder peaks af i stedet for at klippe dem fladt
    return np.tanh(audio_signal / threshold) * threshold


def amplify(audio_signal, target_db=-6.0):
    rms = np.sqrt(np.mean(audio_signal ** 2))
    if rms == 0:
        return audio_signal
    target_rms = 10 ** (target_db / 20)
    gain = target_rms / rms
    amplified = audio_signal * gain
    return soft_limit(amplified)


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
    plt.title("Filtered signal (Chebyshev II + de-emphasis + amplify)")
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
    freqs_o, psd_o = welch(original, fs=sample_rate, nperseg=2048)
    freqs_f, psd_f = welch(filtered, fs=sample_rate, nperseg=2048)

    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs_o, psd_o, label="Original", alpha=0.7)
    plt.semilogy(freqs_f, psd_f, label="Filtered", alpha=0.7)
    plt.title("Power Spectral Density — Original vs Filtered")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.legend()
    plt.tight_layout()
    plt.show()


def remove_blue_noise_from_wav(input_wav_path, output_wav_path, rs=40, target_db=-6.0):
    # Trin 1: Læs lydfil og gør mono
    audio_signal, sample_rate = sf.read(input_wav_path)
    if audio_signal.ndim > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    # Trin 2: Estimér SNR i høj-frekvensbåndet
    snr_db = estimate_snr_highband(audio_signal, sample_rate)
    print(f"Estimeret SNR (høj bånd): {snr_db:.1f} dB")

    # Trin 3: Vælg cutoff og filterorden baseret på SNR
    cutoff_hz, order = select_filter_params(snr_db, sample_rate)
    print(f"Chebyshev II low-pass → cutoff: {cutoff_hz:.0f} Hz, orden: {order}, rs: {rs} dB")

    # Trin 4: De-emphasis — flader blå støjens +6 dB/oktav kurve ud
    deemphasised = apply_deemphasis(audio_signal, strength=0.95)

    # Trin 5: Anvend Chebyshev Type II low-pass filter
    filtered_signal = apply_blue_noise_filter_iir(deemphasised, sample_rate, cutoff_hz, order, rs=rs)

    # Trin 6: Let filter før forstærkning — renser støjrester inden gain
    filtered_signal = apply_gentle_lowpass(filtered_signal, sample_rate, cutoff_hz=8000)

    # Trin 7: Forstærk stemmen med blødt loft mod distortion
    filtered_signal = amplify(filtered_signal, target_db=target_db)
    print(f"Forstærkning → target: {target_db} dB")

    # Trin 8: Let filter efter forstærkning — tager støj løftet af gain
    filtered_signal = apply_gentle_lowpass(filtered_signal, sample_rate, cutoff_hz=8000)

    # Trin 9: Gem resultat
    output_folder = os.path.dirname(output_wav_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    sf.write(output_wav_path, filtered_signal, sample_rate)
    print(f"Gemt filtreret fil: {output_wav_path}")

    # Trin 10: Plot tidssignaler og frekvensrespons
    plot_signals(original=audio_signal, filtered=filtered_signal, sample_rate=sample_rate)
    plot_frequency_response(original=audio_signal, filtered=filtered_signal, sample_rate=sample_rate)


input_path  = "C:\\Users\\TOTAL TECH\\Desktop\\Speach_augmented\\Training\\BlueNoise\\BlueNoise_beta-0.90_snr-0.31_193.wav"
output_path = "C:\\Users\\TOTAL TECH\\Desktop\\Speach_augmented\\Training\\BlueNoise\\Uden\\filtered_blue2.wav"

remove_blue_noise_from_wav(
    input_wav_path=input_path,
    output_wav_path=output_path,
    rs=100,          # Stopbånds-dæmpning — prøv 60 hvis der stadig er støj
    target_db=-7.0  # Forstærkning — prøv -5.0 eller -4.0 hvis stemmen er for stille
)