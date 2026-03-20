import os
import numpy as np
import colorednoise as cn
import soundfile as sf
import random as rand

# -----------------------------
# Settings
# -----------------------------
BASE_DIR = r"C:\Audio\Speach"
OUTPUT_DIR = r"C:\Audio\Speach_augmented"

SPLITS = ['Training', 'Test', 'Validation']

NOISE_RANGES = {
    'WhiteNoise': (-0.1, 0.1),
    'PinkNoise': (0.9, 1.1),
    'BrownNoise': (1.9, 2.1),
    'BlueNoise': (-1.1, -0.9),
    'VioletNoise': (-2.1, -1.9)
}

# -----------------------------
# Noise function
# -----------------------------
def add_colored_noise(signal, beta, snr_db):
    n = len(signal)
    noise = cn.powerlaw_psd_gaussian(beta, n).astype(np.float32)
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    k = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    noisy_signal = signal + noise * k
    return noisy_signal  # no clipping for distortion

# -----------------------------
# Create output directories
# -----------------------------
for split in SPLITS:
    for noise_name in list(NOISE_RANGES.keys()) + ['Clean']:
        out_dir = os.path.join(OUTPUT_DIR, split, noise_name)
        os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# Process dataset
# -----------------------------
for split in SPLITS:
    split_dir = os.path.join(BASE_DIR, split)

    for folder_name in os.listdir(split_dir):
        folder_path = os.path.join(split_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".wav")])

        for idx, file_name in enumerate(files, start=1):
            file_path = os.path.join(folder_path, file_name)

            # -----------------------------
            # Load audio
            # -----------------------------
            signal, samplerate = sf.read(file_path)
            signal = signal.astype(np.float32)

            # -----------------------------
            # Save clean audio
            # -----------------------------
            if folder_name.lower() in ['støjfri', 'clean']:
                out_file_name = f"Clean_{idx:03d}.wav"
                out_path = os.path.join(OUTPUT_DIR, split, "Clean", out_file_name)
                sf.write(out_path, signal, samplerate)
                continue

            # -----------------------------
            # Create noisy versions (one per noise type)
            # -----------------------------
            for noise_name, beta_range in NOISE_RANGES.items():
                beta = rand.uniform(*beta_range)
                snr_db = rand.uniform(-5.0, 20.0)

                noisy_signal = add_colored_noise(signal, beta, snr_db)

                # filename includes beta and SNR
                out_file_name = f"{noise_name}_beta{beta:.2f}_snr{snr_db:.2f}_{idx:03d}.wav"
                out_path = os.path.join(OUTPUT_DIR, split, noise_name, out_file_name)

                sf.write(out_path, noisy_signal, samplerate)

print("All samples processed and saved in:", OUTPUT_DIR)