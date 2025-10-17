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


def create_model(model_source: str, model_type: str = "hf"):
    """Create and cache a model based on the source and type."""
    if model_type == "hf":
        return create_hugging_face_model(model_source)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def create_hugging_face_model(model_name: str):
    print("model name; ", model_name)
    """Create and cache a Hugging Face ASR model pipeline."""
    # Detect available device: CUDA > MPS (Apple) > CPU
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built()
    cuda_available = torch.cuda.is_available()

    # Choose dtype: prefer fp16 on CUDA, otherwise fp32 (MPS doesn't fully support fp16 for many ops)
    torch_dtype = torch.float16 if cuda_available else torch.float32

    # Choose pipeline device: prefer CUDA index, then 'mps' string (transformers may accept it), else 'cpu'
    if cuda_available:
        device_arg = 0
    elif mps_available:
        # Newer transformers versions accept device='mps' or torch.device('mps') for pipelines
        device_arg = "mps"
    else:
        device_arg = "cpu"

    # Create pipeline directly on the chosen device. This avoids having to manually move
    # the internal model tensors later which can produce device-mismatch errors.
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        torch_dtype=torch_dtype,
        device=device_arg,
    )

    return asr


def use_hugging_face_model(model_name: str, input_audio_path: str) -> str:
    """Run ASR on a given audio file using a specified Hugging Face model."""
    asr = create_hugging_face_model(model_name)
    result = asr(input_audio_path)
    return result["text"] if isinstance(result, dict) else result[0]["text"]

def transcribe_with_fallback(model, wav_path) -> str:
  '''
  Given a HuggingFace ASR pipeline model and a path to a .wav file,
  attempt to transcribe the audio using multiple loading methods to
  maximize compatibility across different audio formats and installations.
  Returns the transcribed text or an error message if all methods fail.
  '''
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
          # If torchaudio failed due to missing TorchCodec backend, try soundfile/librosa
          err_msg = str(e2)
          if "TorchCodec is required" in err_msg or "torchcodec" in err_msg:
              try:
                  import soundfile as sf
                  import numpy as _np

                  audio_np, sr = sf.read(str(wav_path))
                  # soundfile returns (samples, channels) for multi-channel
                  if audio_np.ndim > 1:
                      audio_np = audio_np.mean(axis=1)
                  if sr != 16000:
                      import librosa

                      audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
                      sr = 16000
                  out3 = model(audio_np, sampling_rate=sr)
                  return out3["text"] if isinstance(out3, dict) else out3[0]["text"]
              except Exception as e3:
                  hint = (
                      "\nHint: torchaudio reported a TorchCodec requirement. You can install torchcodec\n"
                      "(pip install torchcodec) or install soundfile and librosa (pip install soundfile librosa)\n"
                      "to enable the fallback loader."
                  )
                  return f"ERROR: primary error: {e}; fallback error: {e2}; secondary fallback error: {e3}.{hint}"
          return f"ERROR: primary error: {e}; fallback error: {e2}"
