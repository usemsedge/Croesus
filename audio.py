
from pyexpat import model
import torch
import torchaudio
from transformers import AutoModelForAudioClassification, WhisperFeatureExtractor
import requests
import os
import librosa
from specialized_models import create_model, transcribe_with_fallback, \
  SPECIALIZED_MODELS
import google.generativeai as genai
from dotenv import load_dotenv

from specialized_accent_classifier import predict_audio

LLM_MODEL_ID = "models/gemini-2.5-flash-lite"
ACCENT_PREDICTION_MODEL_ID = "nirmoh/accent-whisper"
MODEL_CACHE = {}

def get_accent_prediction(input_wav_file_path: str, model_id: str = ACCENT_PREDICTION_MODEL_ID):
  """
  Given a .wav path, return accent prediction using either a Transformers audio-classification
  model or (for SpeechBrain-packaged models like the Jzuluaga CommonAccent) fall back to
  SpeechBrain's `foreign_class` interface.

  Returns a dict with either {"accent": label, "confidence": float} or {"error": msg}.
  """

  return predict_audio(input_wav_file_path)

  # If transformers path failed, try SpeechBrain foreign_class for models packaged that way


def get_text_from_speech(input_wav_file_path: str, attributes: dict) -> str:
  '''
  Given a .wav file path and a dictionary of attributes, 
  return the transcribed text using a pretrained ASR model.
  '''
  # Load pretrained ASR model
  # These may be changed based off available specialized models
  gender = attributes.get("gender", None)
  accent = attributes.get("accent", None)

  model = MODEL_CACHE.get(accent)
  if model is None:
    model_source = SPECIALIZED_MODELS.get(accent).get("hf_id")
    if model_source is None:
      model_source = SPECIALIZED_MODELS.get("general").get("hf_id")
    model = create_model(model_source, model_type="hf")
    MODEL_CACHE[accent] = model

  text = transcribe_with_fallback(model, input_wav_file_path)

  return text

def get_llm_response(text: str, attributes: dict) -> str:
  '''
  Given transcribed text and a dictionary of attributes, 
  return a response from a large language model (LLM) with context.
  '''
  # The attributes can be used to provide context to the LLM.
  
  # Example prompt construction

  # Call to LLM API would go here
  load_dotenv()
  genai.configure(api_key=os.getenv("google_api_key"))
  model = genai.GenerativeModel(LLM_MODEL_ID)

  # NOTE: CUSTOMIZE THE PROMPT FOR SPECIFIC USE CASE

  prompt = f"""
  The following text is pulled from an audio file which
  comes from a phone call from a person with the 
  following attributes: {attributes}. 

  The text is part of a story. Add between 30 and 50 words
  to the story.

  Text: {text}
  """
  response = model.generate_content(prompt)
  return response.text

def text_to_speech(text: str, attributes: dict, output_wav_file_path: str) -> None:
  '''
  Given text and a dictionary of attributes, 
  convert the text to speech and save it as a .wav file.
  '''
  # This is a placeholder function. In practice, you would use a TTS library or API.
  
  # Example: Using a hypothetical TTS library
  # tts = SomeTTSLibrary(voice=attributes["accent"],
  raise NotImplementedError("TTS functionality is not implemented yet.")