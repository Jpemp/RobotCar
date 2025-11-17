import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd



#create a seq2seq model

class GRUencoder(nn.Module): #GRU encoder class
    def __init__(self, input_dimension, embed_dimension, hidden_dimension): #class constructor that creates an encoder object
        print("Calling encoder class")
        super.__init__() #used to call the nn.Module parent class and creating a child object that inherits parent attributes, which must be done before creating executing the rest of the child class code
        self.embedding = nn.Embedding(input_dimension, embed_dimension) #creates an embedding table (input dim=number of embeddings that can be put in the embedding table. an embedding is a vector representation of data, like words. embed_dim=size of each embed vector in the table). Embedding is turning words into a numerical vector representation
        self.rnn = nn.GRU(embed_dimension, hidden_dimension) #establishes the gated recurrent unit neural network that will be used to process input data into an output and hidden vector

    def forward(self, input_data): #Whenever data is fed through the encoder object, the forward method is called to perform computations. It feeds the input through nn layers and creates an encoded/hidden vector to return
        print("Calling encoder forward") #seeing if forward pass is called once and RNN is done inside a single method, or if the method is called multiple times for each GRU there is in the encoder
        embedded = self.embedding(input_data) #takes the input data and turns it into embeddings to be stored on the previously created embedding table
        outputs, hidden = self.rnn(embedded) #runs the embeddings through a GRU RNN
        return hidden #returns hidden vector to be used for future GRU layers, with the final hidden vector being sent to the decoder

class GRUdecoder(nn.Module): #GRU decoder class
    def __init__(self, output_dimension, embed_dimension, hidden_dimension):
        print("Calling decoder class")
        super().__init__() #calling nn.Module parent class
        self.embedding = nn.Embedding(output_dimension, embed_dimension) #create an embed table object for the decoder
        self.rnn = nn.GRU (embed_dimension, hidden_dimension) #create a GRU neural network for the decoder to use to translate the hidden vector into translated output data
        self.dl = nn.Linear(hidden_dimension, output_dimension) #creates a linear transform object, input size of hidden_dimension, output size of output_dimension

    def forward(self, h_input_data, hidden):
        print("Calling decoder forward")
        h_input_data = h_input_data.unsqueeze(0) #turns h_input_data tensor into 0-D tensor (an array of numbers)
        embedded = self.embedding(h_input_data) #turns hidden vector from encoder into embedding vectors
        output, hidden = self.rnn(embedded, hidden) #runs embeddings through a GRU neural network
        prediction = self.dl(output.squeeze(0)) #dense layer of linear transformation to put the output through before output/prediction data is complete
        return prediction, hidden #returns prediction(output) data and hidden state to send to future GRU layers. In this case, the output data is the model taking audio and guessing what the audio is saying in a text format

class GRUseq2seq_model(nn.Module): #GRU model class
    def __init__(self, encoder, decoder, device):
        super().__init__() #calling nn.Module parent class
        self.encoder = encoder #class attribute that stores the GRUencoder object
        self.decoder = decoder #class attribute that stores the GRUdecoder object
        self.device = device #class attribute that stores the device object

    def forward(self, input_data, trg, max_length, teacher_forcing_ratio):
        #teacher forcing ratio is the probability that the inputs used for a GRU layer is instead replaced with the target output data as the input instead (output should equal input in this case). This practice is called teacher forcing. This is done so that the model stays stable, makes it more accurate, has faster convergence, and reduces error propogation
        #trg is the target output, which is used for teacher forcing


        batch_size = input_data.shape[1] #number of columns of input_data
        rt_vocab_size = self.decoder.dl.out_features #size of random target vector. Same as size output sample size from linear transformation
        outputs = [] #creating an array to put the model's output sequence in to

        hidden = self.encoder(input_data) #runs the input data through an encoder, outputting the hidden vector

        decoder_input = torch.zeros(batch_size).to(self.device) #creates a tensor with a size of batch size filled with zeros on device chosen. This is will be used as input for GRU decoder units

        for x in range(max_length): #loop decoder function until entire hidden vector sequence has been through the decoder
            output, hidden = self.decoder(decoder_input, hidden) #each output/prediction token is kept and stored in the outputs tensor
            max_ind = output.argmax(1) #the indices where the max value elements across each array in the output token are stored here
            outputs.append(max_ind.unsqueeze(0)) #this is the predicted output sequence, hopefully it is or similar to the target output

            if trg is not None and x < trg.shape[0] and torch.rand(1).item() < teacher_forcing_ratio: #Sometimes the teacher forcing will occur, improving the seq2seq model
                decoder_input = trg[x] #if teacher forcing happens, target output will be the input fed into the next GRU cell
            else:
                decoder_input = max_ind #if teacher forcing doesn't happen, predicted output will be the input fed into the next GRU cell

        outputs = torch.cat(outputs, dim=0) #concatenates all tensors in rows
        return outputs




device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #runs tensors on NVIDIA GPUs, if not, then run tensors on cpu

encoder = GRUencoder() #creates encoder object
decoder = GRUdecoder() #creates decoder object
model = GRUseq2seq_model(encoder, decoder, device).to(device) #creates GRUseq2seq model object which will run tensors on the device selected (CUDA or CPU)

input_data = torch.randint(1) #input data for the model
target_data = torch.randint(1) #target data the model is trying to predict

output_data = model() #data that the model actually predicts

#train it (using my own database if there is time. There will be likely overfitting but it'll be cool to train on my own voice)

#after model is trained, test it for accuracy

#if tests prove the seq2seq model is good, then have the model look for keywords via classification (left, right, speed up, slow down, etc.)
#if keywords are detected, signal sent from raspberryPi to ESP32, either via GPIO pins or bluetooth communication
