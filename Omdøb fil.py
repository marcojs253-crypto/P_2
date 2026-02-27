import os

# ====== INDSTILLINGER ======
mappe = "/Users/jonassvirkaer/Desktop/Datasæt - Støj/Speach/Træning/No Noise"   # <-- Ret denne
nyt_navn = "Lydfil_NoNoise"            # <-- Ret denne

# Find alle wav-filer i mappen
wav_filer = [f for f in os.listdir(mappe) if f.lower().endswith(".wav")]
wav_filer.sort()

for i, fil in enumerate(wav_filer, start=1):
    gammelt_path = os.path.join(mappe, fil)
    
    nyt_filnavn = f"{nyt_navn}_{i:02d}.wav"
    nyt_path = os.path.join(mappe, nyt_filnavn)
    
    os.rename(gammelt_path, nyt_path)
    print(f"Omdøbt: {fil} → {nyt_filnavn}")

print("✅ Færdig!")