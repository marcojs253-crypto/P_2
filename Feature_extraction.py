import sklearn
import matplotlib as plt
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Nødvendige imports til audio feature extraction
import librosa
import numpy as np

# Gemmer datasæt path
datasæt_lyd = Path("/Users/jonassvirkaer/Desktop/python_projekter/P2 - Projekt/NISQA_Corpus/NISQA_TRAIN_LIVE/deg")

# Finder alle .wav-lydfiler i datasæt-mappen (inkl. undermapper) og sorterer dem
alle_wav_files = sorted(datasæt_lyd.rglob("*.wav"))

# Opretter en tom liste, som skal indeholde information om hver lydfil
wav_liste = []
for wav in alle_wav_files:
    # Loader lydfilen (samme samplerate for alle filer)
    y, sr = librosa.load(wav, sr=16000, mono=True)

    # STFT magnitude-spektrogram
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=320))

    # Spectral shape-features (beregnes frame-wise)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    flatness = librosa.feature.spectral_flatness(S=S)

    # Gemmer filnavn (uden path) + aggregerede features
    row = {
        "path": wav.name,   # ← KUN filnavnet

        "centroid_mean": float(np.mean(centroid)),
        "centroid_std":  float(np.std(centroid)),

        "bandwidth_mean": float(np.mean(bandwidth)),
        "bandwidth_std":  float(np.std(bandwidth)),

        "rolloff_mean": float(np.mean(rolloff)),
        "rolloff_std":  float(np.std(rolloff)),

        "flatness_mean": float(np.mean(flatness)),
        "flatness_std":  float(np.std(flatness)),
    }

    wav_liste.append(row)

# Konverterer listen af filnavne + features til en pandas DataFrame
df_files = pd.DataFrame(wav_liste)

# Gemmer DataFrame som CSV
df_files.to_csv("df_files.csv", index=False)