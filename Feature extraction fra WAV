import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from datetime import datetime

datasæt_lyd = Path("/Users/jonassvirkaer/Desktop/Training")
alle_wav_files = sorted(datasæt_lyd.rglob("*.wav"))  # finder også i undermapper

print(f"Finder wav-filer i: {datasæt_lyd}")
print(f"Antal wav-filer fundet: {len(alle_wav_files)}")
if len(alle_wav_files) == 0:
    raise FileNotFoundError("Ingen .wav filer fundet. Tjek path og fil-endelser (.wav/.WAV).")

print("Første 5 filer:")
for w in alle_wav_files[:5]:
    print(" -", w.name)

noise_labels = ["GreenNoise", "BrownNoise", "NoNoise", "WhiteNoise", "PinkNoise"]

wav_liste = []
fejl_filer = []

for wav in alle_wav_files:
    try:
        # --- load ---
        y, sr = librosa.load(wav, sr=16000, mono=True)

        # tom/defekt fil check
        if y is None or len(y) == 0:
            raise ValueError("Tom lyd (0 samples)")

        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=320))

        # --- features ---
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
        flatness = librosa.feature.spectral_flatness(S=S)

        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        flux = librosa.onset.onset_strength(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        y_harmonic = librosa.effects.harmonic(y)
        y_percussive = librosa.effects.percussive(y)
        harmonic_energy = np.sum(y_harmonic ** 2)
        noise_energy = np.sum(y_percussive ** 2)
        hnr = harmonic_energy / (noise_energy + 1e-10)

        # --- label fra filnavn ---
        filename_lower = wav.stem.lower()
        label_found = "Unknown"
        for label in noise_labels:
            if label.lower() in filename_lower:
                label_found = label
                break

        # Debug: hvis den ikke finder label, print filnavnet
        if label_found == "Unknown":
            print(f"⚠️ Ingen label fundet i filnavn: {wav.name}")

        row = {
            "filnavn": wav.name,
            "centroid_mean": float(np.mean(centroid)),
            "centroid_std": float(np.std(centroid)),
            "bandwidth_mean": float(np.mean(bandwidth)),
            "bandwidth_std": float(np.std(bandwidth)),
            "rolloff_mean": float(np.mean(rolloff)),
            "rolloff_std": float(np.std(rolloff)),
            "flatness_mean": float(np.mean(flatness)),
            "flatness_std": float(np.std(flatness)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "flux_mean": float(np.mean(flux)),
            "flux_std": float(np.std(flux)),
            "contrast_mean": float(np.mean(contrast)),
            "contrast_std": float(np.std(contrast)),
            "hnr": float(hnr),
        }

        for i in range(13):
            row[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
            row[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))

        row["target"] = label_found
        wav_liste.append(row)

    except Exception as e:
        fejl_filer.append((wav.name, str(e)))
        print(f"❌ Fejl i {wav.name}: {e}")

df_files = pd.DataFrame(wav_liste)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_path = datasæt_lyd / f"audio_features_dataset_{timestamp}.csv"
df_files.to_csv(output_path, index=False)

print("\n✅ Done!")
print(f"CSV gemt her: {output_path}")
print(f"Antal rækker i CSV: {len(df_files)}")

print("\nLabel-fordeling:")
if len(df_files) > 0:
    print(df_files["target"].value_counts(dropna=False))

print("\nFejl-filer (hvis nogen):", len(fejl_filer))
for name, err in fejl_filer[:10]:
    print(" -", name, "=>", err)