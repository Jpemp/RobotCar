import torch
import torchaudio
from torchaudio.utils import download_asset
import sounddevice as sd
from scipy.io.wavfile import write
import time
import spidev
from pathlib import Path
#import IPython

torch.random.manual_seed(0) #change to test reliability of model and debug. This changes the rng of PyTorch functions
device = torch.device("cpu") #all tensor torch computations occur in the cpu

#SPI bus number
#SPI_bus = 0

#chip select pin. Can be set high(1) or low(0)
#CS_pin = 1

#enables SPI
#spi = spidev.SpiDev()

#opens connection to bus and peripheral (ESP32)
#spi.open(SPI_bus, CS_pin)

#spi.max_speed_hz = 16000 #SPI communication max speed
#spi.mode = 0 #SPI communication mode? There's 4 modes, but 0 is default and shouldn't change

class Decoder(torch.nn.Module): #Class for decoder used to output the results of the Wav2Vec2 model. Also called Greedy Decoder, which means it doesn't use context when determining the current letter being said in the audio, using the highest logit rather than factoring in surrounding letters for predicting the current letter
    def __init__(self, labels, blank): #creates a decoder object
        super().__init__() #calls upon the nn.Module parent class to create the child object that inheirits the parent's attributes
        self.labels = labels #stores the Label array here
        self.blank = blank #index that the "-" label is stored in, which indicates dead air/no talking

    def forward(self, emission): #this function is called whenever a Decoder object is called after object creation
        indices = torch.argmax(emission, dim=-1) #in the emission data, stores the index of the highest logit for each frame. the highest logit is what letter is most likely being said in a frame. each logit index corresponds to the Labels array index (ex. 2 for logit index indicates 'E' in label array, which is indedx 2, is the most likely letter being said) 
        indices = torch.unique_consecutive(indices, dim=-1) #gets rid of consecutive identical numbers in array, which indicates the same letter being said over multiple frames. Gets rid of redundancy
        indices = [i for i in indices if i != self.blank] #gets rid of dead air/"-" indices in the array
        return "".join([self.labels[i] for i in indices]) #creates a string of words based on indices, with each number in indices corresponding to a character in the labels array. This constructed sentence is the likely sentence that was said in the input audio based on the emission logits

#uncomment this section once the sample .wav file is successfully translated by the the program 

#print(sd.query_devices()) #use this to figure out input audio setup

#print("Audio recording starts now")
#my_recording = sd.rec(160000, samplerate=16000, channels=2) #starts recorder to run for 10 seconds at 16kHz (matches ASR model)
#figure out number of channels and available audio devices

#sd.wait() #waits for 10 seconds to record audio before proceeding

#write("mic_input.wav", 16000, my_recording) #creates 16kHz .wav file from microphone audio

#sample = "/home/mic_input.wav" #uses microphone audio to feed through the machine learning model
#print(sample)
#sample = str(sample) #converts audio into the right data type that the model wants

#print(type(sample))
#might not need to use .wav files for wav2vec model. Must test this to find out

sample = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav") #test sample
#print(type(sample))
#print(sample)
#IPython.display.Audio(sample) #displays the .wav file

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_100H #a bundle 100 hours of audio dataset for a wav2vec model, and a Wav2Vec2 model fine tuned for asr

model = bundle.get_model().to(device) #constructs model based on the bundle (a wav2vec model trained on 100 hours of audio, fine tuned for asr)

waveform, sample_rate = torchaudio.load(sample) #audio file's waveform and sample_rate extracted to be used for Wav2Vec2 model

if sample_rate != bundle.sample_rate: #to make sure the input sample_rate matches the model sample rate
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

with torch.inference_mode(): #you can use this to look at acoustic features on the transformer layers
    features, _ = model.extract_features(waveform)

with torch.inference_mode(): #this turns the waveform into a sequence of emissions. An emission is a sequence of logits, which tell the probability that a certain letter is being said in a certain frame of the waveform
    emission, _ = model(waveform)

print(emission)

decoder = Decoder(labels=bundle.get_labels(), blank=0) #creates a Decoder object that uses 
text = decoder(emission[0]) #emission data goes through decoder, outputting text

print(text) #prints out text from decoder to see if its accurate

#confirmation = xfer(text.encode()) #sends text message to ESP32 in the form of bytes, Raspberry Pi recieves confirmation of transfer from ESP32

#confirmation = confirmation.decode() #turns the bytes back into a string

#if confirmation == 1: #confirm that the transfer worked correctly on the Raspberry Pi's end
#    print("SPI transfer succeeded")
#else:
#    print("SPI transfer failed")
