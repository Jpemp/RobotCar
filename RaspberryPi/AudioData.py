import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as FA
import IPython
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

train_dataset = torchaudio.datasets.SPEECHCOMMANDS(root='C:\\data', download=True, subset="training") #downloads audio data from torchaudio onto the computer
test_dataset = torchaudio.datasets.SPEECHCOMMANDS(root='C:\\data', download=True, subset="testing") #split audio data into a train and test dataset

data_loader=DataLoader(train_dataset)
