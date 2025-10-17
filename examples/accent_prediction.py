import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audio import get_accent_prediction

def main():
    k = 10
    samples_dir = Path("samples/ASI/wav")
    if not samples_dir.exists():
        print(f"Samples directory {samples_dir} not found. Create it and add WAVs.")
        return

    wavs = sorted(list(samples_dir.glob("*.wav")))[:k]
    if not wavs:
        print(f"No .wav files found in {samples_dir}")
        return
    print(f"Found {len(wavs)} .wav files in {samples_dir}")


    for wav_path in wavs:
      prediction = get_accent_prediction(wav_path)
      print(f"Accent prediction for {wav_path}: {prediction}")

if __name__ == "__main__":
    main()