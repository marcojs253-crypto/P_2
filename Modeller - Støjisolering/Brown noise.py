import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


# =========================================================
# 1. HIGH-PASS FILTER VIA FFT
# =========================================================
# Denne funktion fjerner lave frekvenser under cutoff_hz.
# Det er nyttigt mod fx brown noise, rumlen og langsom drift.
#
# Parametre:
# - audio_signal: 1D numpy-array med lydsignal
# - sample_rate: samplerate i Hz
# - cutoff_hz: grænsefrekvens i Hz
#
# Returnerer:
# - filtered_signal: high-pass filtreret signal
# =========================================================
def apply_highpass_fft_filter(audio_signal, sample_rate, cutoff_hz):

    # Antal samples i signalet
    n = len(audio_signal)

    # FFT af signalet (kun positive frekvenser)
    spectrum = np.fft.rfft(audio_signal)

    # Frekvensakse til FFT-bin'ene
    freqs = np.fft.rfftfreq(n, d=1 / sample_rate)

    # Maske:
    # False (0) for frekvenser under cutoff
    # True  (1) for frekvenser over cutoff
    mask = freqs >= cutoff_hz

    # Anvend masken i frekvensdomænet
    filtered_spectrum = spectrum * mask

    # Tilbage til tidsdomænet
    filtered_signal = np.fft.irfft(filtered_spectrum, n=n)

    return filtered_signal


# =========================================================
# 2. PLOTFUNKTION
# =========================================================
# Viser:
# 1) original signal
# 2) filtreret signal
# 3) forskellen mellem dem
# =========================================================
def plot_signals(original, filtered, sample_rate):

    # Tidsakse i sekunder
    time = np.arange(len(original)) / sample_rate

    # Den del, som filteret har fjernet
    removed_noise = original - filtered

    plt.figure(figsize=(12, 8))

    # -------- Plot 1: Original --------
    plt.subplot(3, 1, 1)
    plt.plot(time, original)
    plt.title("Original signal")
    plt.xlabel("Time (s)")

    # -------- Plot 2: Filtreret --------
    plt.subplot(3, 1, 2)
    plt.plot(time, filtered)
    plt.title(f"High-pass filtered signal")
    plt.xlabel("Time (s)")

    # -------- Plot 3: Fjernet del --------
    plt.subplot(3, 1, 3)
    plt.plot(time, removed_noise)
    plt.title("Removed part (Original - Filtered)")
    plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


# =========================================================
# 3. HOVEDFUNKTION
# =========================================================
# Step 1: læs WAV-fil
# Step 2: konverter til mono hvis nødvendigt
# Step 3: filtrér med high-pass FFT
# Step 4: opret output-mappe hvis nødvendig
# Step 5: gem outputfil
# Step 6: vis plots
# =========================================================
def remove_brown_noise_from_wav(
    input_wav_path,
    output_wav_path,
    cutoff_hz
):

    # -----------------------------------------------------
    # STEP 1: Læs lydfil
    # -----------------------------------------------------
    audio_signal, sample_rate = sf.read(input_wav_path)

    # -----------------------------------------------------
    # STEP 2: Konverter til mono hvis stereo
    # -----------------------------------------------------
    if audio_signal.ndim > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    # -----------------------------------------------------
    # STEP 3: Anvend high-pass filter
    # -----------------------------------------------------
    filtered_signal = apply_highpass_fft_filter(
        audio_signal,
        sample_rate,
        cutoff_hz
    )

    # -----------------------------------------------------
    # STEP 4: Begræns signalet for at undgå clipping
    # -----------------------------------------------------
    filtered_signal = np.clip(filtered_signal, -1.0, 1.0)

    # -----------------------------------------------------
    # STEP 5: Opret output-mappe hvis nødvendig
    # -----------------------------------------------------
    output_folder = os.path.dirname(output_wav_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # -----------------------------------------------------
    # STEP 6: Gem filtreret lyd
    # -----------------------------------------------------
    sf.write(output_wav_path, filtered_signal, sample_rate)

    # -----------------------------------------------------
    # STEP 7: Plot signalerne
    # -----------------------------------------------------
    plot_signals(
        original=audio_signal,
        filtered=filtered_signal,
        sample_rate=sample_rate
    )


# =========================================================
# 4. FILSTIER
# =========================================================
input_path = "/Users/jonassvirkaer/Desktop/Speach_augmented/Training/BrownNoise/BrownNoise_beta1.90_snr-3.36_168.wav"
output_path = "/Users/jonassvirkaer/Desktop/Uden/filtered.wav"


# =========================================================
# 5. KØR PROGRAMMET
# =========================================================
# Gode testværdier for cutoff_hz:
# 50   -> meget mild
# 100  -> mild
# 200  -> normal
# 300  -> aggressiv
# 500  -> meget aggressiv
# =========================================================
remove_brown_noise_from_wav(
    input_wav_path=input_path,
    output_wav_path=output_path,
    cutoff_hz=200
)