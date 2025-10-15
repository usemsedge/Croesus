import os
import audio


def return_response(input_wav_file_path: str, output_wav_file_path: str) -> dict:
  '''
  test
  '''

  # STEP 0: Check if input file exists and output file does not
  if os.path.exists(input_wav_file_path) == False:
    print("File does not exist")
    return {"status": "error", "message": "File does not exist"}
  if os.path.exists(output_wav_file_path):
    print("Output file already exists")
    return {"status": "error", "message": "Output file already exists"}

  # STEP 1: Get prediction form pretrained accent classifier
  prediction = audio.get_accent_prediction(input_wav_file_path)

  # STEP 2: Get specific model based on prediction
  
  #if prediction is not confident: use general model or ask for clarification

  text = audio.get_text_from_speech(input_wav_file_path, prediction)

  # STEP 3: Give text to LLM with context
  response = audio.get_response_from_LLM(text, prediction)

  # STEP 4: Convert LLM text to speech
  audio.text_to_speech(response, prediction, output_wav_file_path)

  return {
    "status": "success", 
    "file_path": output_wav_file_path, 
    "prediction": prediction
  }