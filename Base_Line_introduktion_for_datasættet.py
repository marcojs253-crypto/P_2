import pandas as pd

df = pd.read_csv("Data/TrainingData.csv")

descriptions = {
    "filnavn":          "Filename of the audio clip",
    "centroid_mean":    "Mean spectral centroid",
    "centroid_std":     "Std. dev. of spectral centroid",
    "bandwidth_mean":   "Mean spectral bandwidth",
    "bandwidth_std":    "Std. dev. of spectral bandwidth",
    "rolloff_mean":     "Mean spectral rolloff",
    "rolloff_std":      "Std. dev. of spectral rolloff",
    "flatness_mean":    "Mean spectral flatness",
    "flatness_std":     "Std. dev. of spectral flatness",
    "zcr_mean":         "Mean zero-crossing rate",
    "zcr_std":          "Std. dev. of zero-crossing rate",
    "rms_mean":         "Mean RMS energy",
    "rms_std":          "Std. dev. of RMS energy",
    "flux_mean":        "Mean spectral flux",
    "flux_std":         "Std. dev. of spectral flux",
    "contrast_mean":    "Mean spectral contrast",
    "contrast_std":     "Std. dev. of spectral contrast",
    "hnr":              "Harmonics-to-Noise Ratio",
    "target":           "Target variable: instrument class",
    "beta":             "Beta shape parameter (345 missing)",
    "snr_db":           "Signal-to-Noise Ratio in dB (345 missing)",
    "mfcc1_mean":       "Mean of MFCC coefficient 1",
    "mfcc1_std":        "Std. dev. of MFCC coefficient 1",
    "mfcc2_mean":       "Mean of MFCC coefficient 2",
    "mfcc2_std":        "Std. dev. of MFCC coefficient 2",
    "mfcc3_mean":       "Mean of MFCC coefficient 3",
    "mfcc3_std":        "Std. dev. of MFCC coefficient 3",
    "mfcc4_mean":       "Mean of MFCC coefficient 4",
    "mfcc4_std":        "Std. dev. of MFCC coefficient 4",
    "mfcc5_mean":       "Mean of MFCC coefficient 5",
    "mfcc5_std":        "Std. dev. of MFCC coefficient 5",
    "mfcc6_mean":       "Mean of MFCC coefficient 6",
    "mfcc6_std":        "Std. dev. of MFCC coefficient 6",
    "mfcc7_mean":       "Mean of MFCC coefficient 7",
    "mfcc7_std":        "Std. dev. of MFCC coefficient 7",
    "mfcc8_mean":       "Mean of MFCC coefficient 8",
    "mfcc8_std":        "Std. dev. of MFCC coefficient 8",
    "mfcc9_mean":       "Mean of MFCC coefficient 9",
    "mfcc9_std":        "Std. dev. of MFCC coefficient 9",
    "mfcc10_mean":      "Mean of MFCC coefficient 10",
    "mfcc10_std":       "Std. dev. of MFCC coefficient 10",
    "mfcc11_mean":      "Mean of MFCC coefficient 11",
    "mfcc11_std":       "Std. dev. of MFCC coefficient 11",
    "mfcc12_mean":      "Mean of MFCC coefficient 12",
    "mfcc12_std":       "Std. dev. of MFCC coefficient 12",
    "mfcc13_mean":      "Mean of MFCC coefficient 13",
    "mfcc13_std":       "Std. dev. of MFCC coefficient 13",
}

print(f"{'#':<4} {'Column':<22} {'Non-Null Count':<16} {'Dtype':<10} {'Description'}")
print("-" * 80)

for i, col in enumerate(df.columns):
    non_null = df[col].notnull().sum()
    dtype    = str(df[col].dtype)
    desc     = descriptions.get(col, "")
    print(f"{i:<4} {col:<22} {non_null:<16} {dtype:<10} {desc}")

print()
print(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")
