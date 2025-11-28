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

with torch.inference_mode(): #this turns the waveform into a sequence of emissions. An emission is a sequence of logits, which tell the probability that a certain letter is being said in a certain frame of the waveform
    emission, _ = model(waveform)

print(emission)

class Decoder(torch.nn.Module): #Class for decoder (called Greed
    def __init__(self, labels, blank): #creates a decoder object
        super().__init__() #calls upon the nn.Module parent class to create the child object that inheirits the parent's attributes
        self.labels = labels #stores the Label array here
        self.blank = blank #index that the "-" label is stored in, which indicates dead air/no talking

    def forward(self, emission): #this function is called whenever a Decoder object is called after object creation
        indices = torch.argmax(emission, dim=-1) #in the emission data, stores the index of the highest logit for each frame. the highest logit is what letter is most likely being said in a frame. each logit index corresponds to the Labels array index (ex. 2 for logit index indicates 'E' in label array, which is indedx 2, is the most likely letter being said) 
        indices = torch.unique_consecutive(indices, dim=-1) #gets rid of consecutive identical numbers in array, which indicates the same letter being said over multiple frames. Gets rid of redundancy
        indices = [i for i indices if i != self.blank] #gets rid of dead air/"-" indices in the array
        return "".join([self.labels[i] for i in indices]) #creates a string of words based on indices, with each number in indices corresponding to a character in the labels array. This constructed sentence is the likely sentence that was said in the input audio based on the emission logits

decoder = Decoder(labels=bundle.get_labels(), blank=0) #creates a Decoder object that uses 
text = decoder(emission[0]) #emission data goes through decoder, outputting text

print(text) #prints out text from decoder to see if its accurate
