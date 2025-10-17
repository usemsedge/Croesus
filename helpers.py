


from pathlib import Path
import sounddevice as sd
from scipy.io.wavfile import write, read

def read_transcript(wav_path: Path):
    # Try several locations for transcripts:
    # 1) same directory as wav (basename.txt)
    # 2) parent 'transcript' directory (e.g., samples/ASI/transcript)
    # 3) any transcript/ directory under samples/**/transcript
    candidates = []
    candidates.append(wav_path.with_suffix('.txt'))

    # parent-level transcript folder (e.g., samples/ASI/transcript)
    try:
        parent = wav_path.parent
        grandparent = parent.parent
        candidates.append(grandparent / 'transcript' / (wav_path.stem + '.txt'))
    except Exception:
        pass

    # generic samples transcript folders
    samples_root = Path('samples')
    if samples_root.exists():
        for tdir in samples_root.rglob('transcript'):
            candidates.append(tdir / (wav_path.stem + '.txt'))

    # check candidates in order
    for c in candidates:
        if c and c.exists():
            try:
                return c.read_text(encoding='utf-8').strip()
            except Exception:
                # if reading fails, skip
                continue

    return None

def wer(ref: str, hyp: str) -> float:
    # simple word-level WER using Levenshtein
    r = ref.split()
    h = hyp.split()
    # build matrix
    import numpy as _np

    d = _np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    if len(r) == 0:
        return float(len(h))
    return float(d[len(r)][len(h)]) / float(len(r))

def cer(ref: str, hyp: str) -> float:
    # character error rate (simple)
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    import numpy as _np

    d = _np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    if len(r) == 0:
        return float(len(h))
    return float(d[len(r)][len(h)]) / float(len(r))


def record_audio(file_path: str, time_seconds: int):
    # Settings
    duration = time_seconds  # seconds
    sample_rate = 16000  # 16 kHz, good for speech

    print(f"🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # wait until recording is done
    print("✅ Done!")

    # Save to a WAV file
    write(file_path, sample_rate, audio)

def play_audio(file_path: str):

    # Load and play
    sample_rate, data = read(file_path)
    print("🎧 Playing...")
    sd.play(data, sample_rate)
    sd.wait()  # Wait until playback finishes
    print("✅ Done!")