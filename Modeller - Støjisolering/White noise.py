import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


# =========================================================
# 1. FILTERFUNKTION
# =========================================================
# Denne funktion laver et simpelt moving average FIR-filter.
#
# Idé:
# - Vi laver et vindue med ens koefficienter
# - Hver outputværdi bliver gennemsnittet af naboværdier
# - Det glatter signalet og dæmper hurtige variationer
#
# Returnerer:
# - filtered_signal: det filtrerede signal
# =========================================================
def apply_moving_average_fir_filter(audio_signal, window_length):

    # FIR-koefficienter:
    # np.ones(window_length) laver fx [1, 1, 1, 1, 1]
    # Divideret med window_length giver et gennemsnitsfilter
    fir_coefficients = np.ones(window_length) / window_length

    # np.convolve anvender filteret på signalet
    # mode="same" betyder:
    # output får samme længde som input
    filtered_signal = np.convolve(
        audio_signal,
        fir_coefficients,
        mode="same"
    )

    return filtered_signal


# =========================================================
# 2. PLOTFUNKTION
# =========================================================
# Denne funktion viser:
# 1) det originale signal
# 2) det filtrerede signal
# 3) forskellen mellem dem (det "fjernede")
# =========================================================
def plot_signals(original, filtered, sample_rate):

    # Lav tidsakse i sekunder
    time = np.arange(len(original)) / sample_rate

    # Forskellen mellem original og filtreret signal
    # Dette tolkes som den del filteret har fjernet
    removed_noise = original - filtered

    # Opret figur
    plt.figure(figsize=(12, 8))

    # -------- Plot 1: Original signal --------
    plt.subplot(3, 1, 1)
    plt.plot(time, original)
    plt.title("Original signal")
    plt.xlabel("Time (s)")

    # -------- Plot 2: Filtreret signal --------
    plt.subplot(3, 1, 2)
    plt.plot(time, filtered)
    plt.title("Filtered signal")
    plt.xlabel("Time (s)")

    # -------- Plot 3: Fjernet del --------
    plt.subplot(3, 1, 3)
    plt.plot(time, removed_noise)
    plt.title("Removed noise (Original - Filtered)")
    plt.xlabel("Time (s)")

    # Gør layout pænere
    plt.tight_layout()
    plt.show()


# =========================================================
# 3. HOVEDFUNKTION: LÆS, FILTRÉR, GEM, VIS
# =========================================================
# Denne funktion styrer hele processen:
#
# Step 1: læs WAV-fil
# Step 2: gør signal mono hvis det er stereo
# Step 3: filtrér signalet
# Step 4: opret output-mappe hvis nødvendig
# Step 5: gem resultatet som WAV
# Step 6: plot signalerne
# =========================================================
def remove_white_noise_from_wav(
    input_wav_path,
    output_wav_path,
    window_length
):

    # -----------------------------------------------------
    # STEP 1: Læs lydfilen
    # -----------------------------------------------------
    # sf.read returnerer:
    # - audio_signal: lyddata som numpy-array
    # - sample_rate: samplerate, fx 16000 eller 44100
    # =========================================================
    audio_signal, sample_rate = sf.read(input_wav_path)

    # -----------------------------------------------------
    # STEP 2: Konverter til mono hvis nødvendigt
    # -----------------------------------------------------
    # Hvis signalet har flere kanaler (fx stereo),
    # tager vi gennemsnittet over kanalerne.
    # =========================================================

    if audio_signal.ndim > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    # -----------------------------------------------------
    # STEP 3: Anvend moving average FIR-filter
    # -----------------------------------------------------
    filtered_signal = apply_moving_average_fir_filter(
        audio_signal,
        window_length
    )

    # -----------------------------------------------------
    # STEP 4: Opret output-mappe hvis den ikke findes
    # -----------------------------------------------------
    # os.path.dirname henter mappen fra output-stien
    # -----------------------------------------------------
    output_folder = os.path.dirname(output_wav_path)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # -----------------------------------------------------
    # STEP 5: Gem det filtrerede signal som WAV-fil
    # -----------------------------------------------------
    sf.write(output_wav_path, filtered_signal, sample_rate)

    # -----------------------------------------------------
    # STEP 6: Vis plots af signalerne
    # -----------------------------------------------------
    plot_signals(
        original=audio_signal,
        filtered=filtered_signal,
        sample_rate=sample_rate
    )


# =========================================================
# 4. FILSTIER
# =========================================================
# - Hvilken fil der skal læses
# - Hvor resultatet skal gemmes
# =========================================================
input_path = "/Users/jonassvirkaer/Desktop/Speach_augmented/Training/WhiteNoise/WhiteNoise_beta-0.03_snr18.03_108.wav"

output_path = "/Users/jonassvirkaer/Desktop/Uden/filtered.wav"


# =========================================================
# 5. KØR PROGRAMMET
# =========================================================
# Her kaldes hovedfunktionen.
#
# window_length=9 betyder:
# - filteret tager gennemsnittet af 9 samples ad gangen
# - større værdi = mere glatning
# - mindre værdi = mindre glatning
# =========================================================
remove_white_noise_from_wav(
    input_wav_path=input_path,
    output_wav_path=output_path,
    window_length=9
)