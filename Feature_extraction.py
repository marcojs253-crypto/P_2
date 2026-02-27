import sklearn
import matplotlib as plt
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Nødvendige imports til audio feature extraction
import librosa
import numpy as np

# Gemmer datasæt path
datasæt_lyd = Path("/Users/jonassvirkaer/Desktop/Training/")

# Loader MOS-scores fra CSV fil
csv_path = Path(r"/Users/jonassvirkaer/Desktop/python_projekter/P2 - Projekt/NISQA_Corpus/NISQA_TRAIN_LIVE/NISQA_TRAIN_LIVE_con.csv")
df_mos = pd.read_csv(csv_path)

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
    # Ekstraktion af con-nummeret fra filnavnet (f.eks. "deg_001.wav" -> 1)
    filename = wav.stem  # får filnavnet uden extension
    try:
        con_number = int(filename.split('_')[-1])
    except (ValueError, IndexError):
        con_number = None
    
    # Finder MOS-score i CSV hvis con-nummer kunne ekstraeres
    mos_score = None
    if con_number is not None:
        matching_row = df_mos[df_mos['con'] == con_number]
        if not matching_row.empty:
            mos_score = matching_row.iloc[0]['mos']

    # Gemmer filnavn + aggregerede features + MOS-score så hver fil bliver én række
    

    # Gemmer filnavn (uden path) + aggregerede features
    row = {
        "filnavn": wav.name,

        "centroid_mean": float(np.mean(centroid)),
        "centroid_std":  float(np.std(centroid)),

        "bandwidth_mean": float(np.mean(bandwidth)),
        "bandwidth_std":  float(np.std(bandwidth)),

        "rolloff_mean": float(np.mean(rolloff)),
        "rolloff_std":  float(np.std(rolloff)),

        "flatness_mean": float(np.mean(flatness)),
        "flatness_std":  float(np.std(flatness)),

        "mos": mos_score,
    }

    wav_liste.append(row)

# Konverterer listen af filnavne + features til en pandas DataFrame
df_files = pd.DataFrame(wav_liste)

# Gemmer DataFrame som CSV med timestamp (overskriver ikke)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
df_files.to_csv(f"df_files_{timestamp}.csv", index=False)