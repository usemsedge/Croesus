import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor, WhisperModel
import numpy as np
from safetensors.torch import load_file

# Define the model class (same as training)
class WhisperClassifier(nn.Module):
    def __init__(self, model_name="openai/whisper-small.en", num_accent_classes=23, num_gender_classes=2, 
                 freeze_encoder=True, dropout_rate=0.3):
        super().__init__()
        
        self.whisper = WhisperModel.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.whisper.encoder.parameters():
                param.requires_grad = False
                
        self.hidden_size = self.whisper.config.d_model
        self.dropout = nn.Dropout(dropout_rate)
        
        # Accent classification head
        
        self.accent_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_accent_classes)
        )
        
        '''
        # Gender classification head
        self.gender_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_gender_classes)
        )
        '''
        self.num_accent_classes = num_accent_classes
        #self.num_gender_classes = num_gender_classes
        
    def forward(self, input_features, accent_labels=None, gender_labels=None):
        encoder_outputs = self.whisper.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        
        accent_logits = self.accent_classifier(pooled_output)
        #gender_logits = self.gender_classifier(pooled_output)
        
        return {
            'accent_logits': accent_logits,
            #'gender_logits': gender_logits,
        }

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WhisperClassifier()

# Load the trained weights
path = "/Users/owenfei/.cache/huggingface/hub/models--nirmoh--accent-whisper/snapshots/70e4d1085f48e6ccdc8efa0b1c03864133cfa212/accent_classifier.safetensors"
state = load_file(path, device="cpu")   # or device="mps"/"cuda"
# inspect keys if you want:
print(list(state.keys())[:40])

model.load_state_dict(state, strict=False)
model.to(device)
model.eval()

# Initialize feature extractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small.en")
def predict_audio(audio_file_path, \
                  model = model, \
                    feature_extractor = feature_extractor, \
                      device = device):
    """
    Predict accent and gender from an audio file
    
    Args:
        audio_file_path: Path to audio file (.wav, .mp3, etc.)
        model: Trained WhisperClassifier model
        feature_extractor: Whisper feature extractor
        device: torch device (cuda/cpu)
    
    Returns:
        Dictionary with predictions and confidence scores
    """
    import librosa
    
    # Load audio file
    audio, sr = librosa.load(audio_file_path, sr=16000, mono=True)
    
    # Extract features
    inputs = feature_extractor(
        audio, 
        sampling_rate=sr, 
        return_tensors="pt"
    )
    
    # Move to device
    input_features = inputs.input_features.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_features=input_features)
        
        # Get probabilities
        accent_probs = F.softmax(outputs["accent_logits"], dim=-1)
        #gender_probs = F.softmax(outputs["gender_logits"], dim=-1)
      
        
        # Get predictions
        accent_pred = torch.argmax(accent_probs, dim=-1).item()
        #gender_pred = torch.argmax(gender_probs, dim=-1).item()
        
        # Get confidence scores
        accent_confidence = accent_probs[0, accent_pred].item()
        #gender_confidence = gender_probs[0, gender_pred].item()
    
    
    # Map predictions to labels
    accent_names = [
        "dutch",
        "german",
        "czech",
        "polish",
        "french",
        "hungarian",
        "finnish",
        "romanian",
        "slovak",
        "spanish",
        "italian",
        "estonian",
        "lithuanian",
        "croatian",
        "slovene",
        "english",
        "scottish",
        "irish",
        "northernirish",
        "indian",
        "vietnamese",
        "canadian",
        "american"
    ]
    
    accent_name = accent_names[accent_pred] if accent_pred < len(accent_names) else f"accent_{accent_pred}"
    #gender_name = "male" if gender_pred == 0 else "female"
    
    return {
        'accent': accent_name,
        'accent_confidence': accent_confidence,
        #'gender': gender_name,
        #'gender_confidence': gender_confidence
    }

# Example usage
if __name__ == "__main__":
  for i in range(1, 100):
      result = predict_audio(f"./samples/ASI/wav/arctic_a{i:04d}.wav", model, feature_extractor, device)
      if result['accent_confidence'] < 0.3:
          print(f"Low confidence ({result['accent_confidence']:.3f}) for file arctic_a{i:04d}.wav")
      else:
          print(f"Predicted Accent: {result['accent']} (confidence: {result['accent_confidence']:.3f})")
          #print(f"Predicted Gender: {result['gender']} (confidence: {result['gender_confidence']:.3f})")