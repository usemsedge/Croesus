import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio import get_text_from_speech
from specialized_models import create_hugging_face_model,\
  SPECIALIZED_MODELS
from helpers import read_transcript as _read_transcript, \
  wer as _wer, \
  cer as _cer
import time

import string

def remove_punct_and_lower(s: str) -> str:
    """
    Remove ASCII punctuation characters and lowercase the string.

    Example:
    >>> remove_punct_and_lower("Hello, World!!")
    'hello world'
    """
    trans = str.maketrans('', '', string.punctuation)
    return s.translate(trans).lower()

def main():
    # Simple runner:
    # - find up to k wav files under samples/ASI/wav
    # - for each: get accent prediction, transcribe with general and Indian models
    # - if a transcript file with same basename + .txt exists, compute simple WER/CER
    k = 100
    samples_dir = Path("samples/ASI/wav")
    if not samples_dir.exists():
        print(f"Samples directory {samples_dir} not found. Create it and add WAVs.")
        return

    wavs = sorted(list(samples_dir.glob("*.wav")))[:k]
    if not wavs:
        print(f"No .wav files found in {samples_dir}")
        return
    print(f"Found {len(wavs)} .wav files in {samples_dir}")


    #gen_model = create_hugging_face_model(SPECIALIZED_MODELS["general"]["hf_id"])
    #ind_model = create_hugging_face_model(SPECIALIZED_MODELS["indian"]["hf_id"])

    # Run
    wer_proportional_diffs = []
    cer_proportional_diffs = []
    wer_absolute_diffs = []
    cer_absolute_diffs = []
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

        start = time.time()
        ind_text = get_text_from_speech(str(wav), {"accent": "indian"})

        mid = time.time()
        print("Indian model infer sec:", mid - start)
        gen_text = get_text_from_speech(str(wav), {"accent": "general"})
        end = time.time()
        print("General model infer sec:", end - mid)

        print(f"Indian model transcription: {ind_text}")
        print(f"General model transcription: {gen_text}")

        if transcript:
            transcript_p = remove_punct_and_lower(transcript)
            ind_text_p = remove_punct_and_lower(ind_text)
            gen_text_p = remove_punct_and_lower(gen_text)
            w_ind = _wer(transcript_p, ind_text_p)
            w_gen = _wer(transcript_p, gen_text_p)
            c_ind = _cer(transcript_p, ind_text_p)
            c_gen = _cer(transcript_p, gen_text_p)
            '''
            w_ind = _wer(transcript.lower(), ind_text.lower())
            w_gen = _wer(transcript.lower(), gen_text.lower())
            c_ind = _cer(transcript.lower(), ind_text.lower())
            c_gen = _cer(transcript.lower(), gen_text.lower())'''

            print(f"WER Indian: {w_ind:.3f}, General: {w_gen:.3f}")
            print(f"CER Indian: {c_ind:.3f}, General: {c_gen:.3f}")
            wer_proportional_diffs.append(w_gen / (w_ind + 1e-6))
            cer_proportional_diffs.append(c_gen / (c_ind + 1e-6))
            wer_absolute_diffs.append(w_gen - w_ind)
            cer_absolute_diffs.append(c_gen - c_ind)
        else:
            print("No reference transcript found (basename.txt) — skipping WER/CER")

    print("Done.")
    if wer_proportional_diffs:
        print("--- Detailed diffs per sample ---")
        print(wer_proportional_diffs)
        print(cer_proportional_diffs)
        print(wer_absolute_diffs)
        print(cer_absolute_diffs)
        print("--- Summary over all samples ---")
        print("Ratio")
        print("(gen / ind) = how many more times errors general model made vs Indian")
        print("A (gen / ind)  > 1 indicates a better ind")
        print(f"Avg proportional WER diff (gen / ind) over {len(wer_proportional_diffs)} samples: "
              f"{sum(wer_proportional_diffs)/len(wer_proportional_diffs):.3f}")
        print(f"Avg proportional CER diff (gen / ind) over {len(cer_proportional_diffs)} samples: "
              f"{sum(cer_proportional_diffs)/len(cer_proportional_diffs):.3f}")
        print("Absolute")
        print("(gen - ind) = how many more absolute errors general model made vs Indian")
        print("A positive absolute (gen - ind) indicates a better ind")
        print(f"Avg absolute WER diff (gen - ind) over {len(wer_absolute_diffs)} samples: "
              f"{sum(wer_absolute_diffs)/len(wer_absolute_diffs):.3f}")
        print(f"Avg absolute CER diff (gen - ind) over {len(cer_absolute_diffs)} samples: "
              f"{sum(cer_absolute_diffs)/len(cer_absolute_diffs):.3f}")
if __name__ == "__main__":
    main()