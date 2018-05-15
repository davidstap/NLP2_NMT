from __future__ import unicode_literals, print_function, division
from utils import load_data, preprocess, Lang, MAX_LENGTH
from models import AttnDecoderRNN, EncoderRNN, DecoderRNN,EncoderPositional
from training import trainIters
from io import open
import unicodedata
import string
import re
import random

# Pytorch imports
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load corpus
input_lang, output_lang, pairs = load_data('train', 'lowercasing')

# Define encoder/decoder model
word_embed_size = 256
pos_embed_size = 20
hidden_size = word_embed_size + pos_embed_size
max_length = 50

# encoder = EncoderPositionalSimple(input_lang.n_words, hidden_size).to(device)
# encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
encoder = EncoderPositional(input_lang.n_words, word_embed_size, pos_embed_size).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

# Train model
n_iters = 75000
trainIters(input_lang, output_lang, pairs, encoder, attn_decoder, n_iters, print_every=50)
