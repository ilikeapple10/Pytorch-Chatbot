
import torch
import torch.nn as nn
from torch import optim
import Vocabulary
import NNetwork
import os

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class model(nn.Module):
    def __init__(self, corpus, datafile, path=""):
        super(model, self).__init__()

        print('Building encoder and decoder ...')
        self.model_name = 'cb_model'
        self.attn_model = 'dot'
        #attn_model = 'general'
        #attn_model = 'concat'
        self.hidden_size = 1000
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.1
        self.batch_size = 64
        self.learning_rate = 0.0001
        self.decoder_learning_ratio = 5.0
        self.datafile = datafile
        self.corpus = corpus
        self.voc, self.pairs = Vocabulary.loadPrepareData(os.path.join("data", corpus), datafile)
        self.embedding = nn.Embedding(self.voc.num_words, self.hidden_size)
        # Initialize encoder & decoder models
        encoder = NNetwork.EncoderRNN(self.hidden_size, self.embedding, self.encoder_n_layers, self.dropout)
        decoder = NNetwork.LuongAttnDecoderRNN(self.attn_model, self.embedding, self.hidden_size, self.voc.num_words, self.decoder_n_layers, self.dropout)
        self.encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(decoder.parameters(), lr=self.learning_rate * self.decoder_learning_ratio)
        # Use appropriate device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        if path != "":
            model_path = torch.load(path)
            super(model, self).__init__()
            self.model_name = model_path['model']
            self.attn_model = model_path['attn']
            self.hidden_size = model_path['hidden']
            self.encoder_sd = model_path['en']
            self.encoder_n_layers = model_path['en_l']
            self.decoder_sd = model_path['de']
            self.decoder_n_layers = model_path['de_l']
            self.encoder_optimizer_sd = model_path['en_opt']
            self.decoder_optimizer_sd = model_path['de_opt']
            self.embedding_sd = model_path['embedding']
            self.voc = Vocabulary.Voc(corpus)
            self.voc.__dict__ = model_path['voc_dict']
            self.iteration = model_path['iteration']
            print('Building encoder and decoder ...')
            # Initialize word embeddings
            self.embedding = nn.Embedding(self.voc.num_words, self.hidden_size)
            #if loadFilename:
            self.embedding.load_state_dict(self.embedding_sd)
            # Initialize encoder & decoder models
            encoder = NNetwork.EncoderRNN(self.hidden_size, self.embedding, self.encoder_n_layers, self.dropout)
            decoder = NNetwork.LuongAttnDecoderRNN(self.attn_model, self.embedding, self.hidden_size, self.voc.num_words, self.decoder_n_layers, self.dropout)
            #if loadFilename:
            encoder.load_state_dict(self.encoder_sd)
            decoder.load_state_dict(self.decoder_sd)
            self.encoder_optimizer.load_state_dict(self.encoder_optimizer_sd)
            self.decoder_optimizer.load_state_dict(self.decoder_optimizer_sd)
            # Use appropriate device
            self.encoder = encoder.to(device)
            self.decoder = decoder.to(device)
            print('Models built and ready to go! (with path)')
        else:
            print('Models built and ready to go! (no path)')