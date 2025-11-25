import torch
from transformers import Speech2TextProcessor
from datasets import load_dataset
import torchaudio

from torchaudio.utils import _download_asset

torch.random.manual_seed(0) #change to test reliability of model and debug. This changes the rng of PyTorch functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample = _download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav") #test sample

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_100H #a bundle 100 hours of audio dataset for a wav2vec model, and a wav2vec model fine tuned for asr

model = bundle.get_model().to(device) #constructs model based on the bundle (a wav2vec model trained on 100 hours of audio, fine tuned for asr)

waveform, sample_rate = torchaudio.load(sample)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate: #to make sure the input sample_rate matches the model sample rate
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

with torch.inference_mode(): #you can use this to look at acoustic features on the transformer layers
    features, _ = model.extract_features(waveform)

with torch.inference_mode():
    emission, _ = model(waveform)

print(emission)

class Decoder(torch.nn.Module):
    def __init__(self, labels, blank):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission):
        print("Test")
