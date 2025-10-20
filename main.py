from examples.return_response import return_response
from helpers import record_audio, play_audio

if __name__ == "__main__":
    record_audio("input.wav", 5)
    result = return_response("input.wav", "output.wav")