from transformers import pipeline
import torch

# Accent-specialized models
"""Specialized model registry.

Add HF model ids under each language as the value for key 'hf_id' if you
want the automated comparison runner (main) to load and run them.

Example:
SPECIALIZED_MODELS = {
    "english_commonaccent": {"hf_id": "openai/whisper-small"},
    "chinese": {"hf_id": ""},
}
"""

from typing import Dict, Any
import os
import requests
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# not using openai/whisper-large-v3, its slow
# will set up gpu acceleration eventually
SPECIALIZED_MODELS: Dict[str, Dict[str, Any]] = {
    "indian": {"hf_id": "Tejveer12/Indian-Accent-English-Whisper-Finetuned"},
    "singapore": {"hf_id": "mjwong/whisper-large-v3-turbo-singlish"},
    "hongkong": {"hf_id": "openai/whisper-small"},
    # Example: uncomment and set an HF id to include in automated runs
    "general": {"hf_id": "openai/whisper-small"},
}


# Cache loaded models so you don’t reload for every request
MODEL_CACHE = {}


def create_hugging_face_model(model_name: str):
    """Create and cache a Hugging Face ASR model pipeline."""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=0 if torch.cuda.is_available() else "cpu",
    )

    MODEL_CACHE[model_name] = asr
    return asr


def use_hugging_face_model(model_name: str, input_audio_path: str) -> str:
    """Run ASR on a given audio file using a specified Hugging Face model."""
    asr = create_hugging_face_model(model_name)
    result = asr(input_audio_path)
    return result["text"] if isinstance(result, dict) else result[0]["text"]


def transcribe_by_accent(accent: str, input_audio_path: str) -> str:
    """Select an accent-specific ASR model and transcribe the given audio."""
    accent = accent.lower().strip()
    if accent not in SPECIALIZED_MODELS:
        raise ValueError(f"No specialized model found for accent '{accent}'")
    model_info = SPECIALIZED_MODELS[accent]
    hf_id = model_info.get("hf_id") or model_info.get("model_name")
    if not hf_id:
        raise ValueError(f"No HF model id configured for accent '{accent}'")
    print(f"Using {hf_id} for accent '{accent}'...")
    text = use_hugging_face_model(hf_id, input_audio_path)
    return text


def main():
    # Simple runner:
    # - find up to 10 wav files under samples/ASI/wav
    # - for each: get accent prediction, transcribe with general and Indian models
    # - if a transcript file with same basename + .txt exists, compute simple WER/CER
    import time
    import audio
    from pathlib import Path

    samples_dir = Path("samples/ASI/wav")
    if not samples_dir.exists():
        print(f"Samples directory {samples_dir} not found. Create it and add WAVs.")
        return

    wavs = sorted(list(samples_dir.glob("*.wav")))[:10]
    if not wavs:
        print(f"No .wav files found in {samples_dir}")
        return
    print(f"Found {len(wavs)} .wav files in {samples_dir}")

    # Prepare models
    # general model for comparison
    general_hf = "openai/whisper-small"
    indian_hf = SPECIALIZED_MODELS.get("indian", {}).get("hf_id")
    if not indian_hf:
        print("No Indian model configured in SPECIALIZED_MODELS['indian']. Please set 'hf_id'.")
        return

    print(f"Loading general model {general_hf} and Indian model {indian_hf} (cached)")
    gen_model = create_hugging_face_model(general_hf)
    ind_model = create_hugging_face_model(indian_hf)

    def _read_transcript(wav_path: Path):
        txt = wav_path.with_suffix(".txt")
        if txt.exists():
            return txt.read_text(encoding="utf-8").strip()
        return None

    def _wer(ref: str, hyp: str) -> float:
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

    def _cer(ref: str, hyp: str) -> float:
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

    # Run
    for wav in wavs:
        print("---")
        print(f"Sample: {wav}")
        transcript = _read_transcript(wav)

        '''
        # Accent prediction (uses audio.get_accent_prediction)
        try:
            accent_pred = audio.get_accent_prediction(str(wav))
        except Exception as e:
            accent_pred = {"error": str(e)}
        print(f"Accent prediction: {accent_pred}");'''

        # Preferred: pass filename so HF pipeline can use ffmpeg for decoding
        # If that fails (ffmpeg missing or pipeline error), fall back to loading
        # the audio into numpy and passing arrays to the pipeline.
        def _transcribe_with_fallback(model, wav_path):
            # try filename first
            try:
                out = model(str(wav_path))
                return out["text"] if isinstance(out, dict) else out[0]["text"]
            except Exception as e:
                # fallback to numpy approach
                try:
                    waveform, sr = torchaudio.load(str(wav_path))
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0)
                    audio_np = waveform.squeeze().numpy()
                    if sr != 16000:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                        audio_np = resampler(torch.tensor(audio_np)).numpy()
                        sr = 16000
                    out2 = model(audio_np, sampling_rate=sr)
                    return out2["text"] if isinstance(out2, dict) else out2[0]["text"]
                except Exception as e2:
                    return f"ERROR: primary error: {e}; fallback error: {e2}"
        start = time.time()
        ind_text = _transcribe_with_fallback(ind_model, wav)
        mid = time.time()
        print("Indian model infer sec:", mid - start)
        gen_text = _transcribe_with_fallback(gen_model, wav)
        end = time.time()
        print("General model infer sec:", end - mid)

        print(f"Indian model transcription: {ind_text}")
        print(f"General model transcription: {gen_text}")

        if transcript:
            w_ind = _wer(transcript.lower(), ind_text.lower())
            w_gen = _wer(transcript.lower(), gen_text.lower())
            c_ind = _cer(transcript.lower(), ind_text.lower())
            c_gen = _cer(transcript.lower(), gen_text.lower())
            print(f"WER Indian: {w_ind:.3f}, General: {w_gen:.3f}")
            print(f"CER Indian: {c_ind:.3f}, General: {c_gen:.3f}")
        else:
            print("No reference transcript found (basename.txt) — skipping WER/CER")

    print("Done.")

if __name__ == "__main__":
    main()