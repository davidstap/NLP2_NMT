from __future__ import unicode_literals, print_function, division
from utils import load_data, preprocess, Lang, MAX_LENGTH
from models import AttnDecoderRNN, EncoderRNN, DecoderRNN
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
hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

# Train model
n_iters = 75000
trainIters(input_lang, output_lang, pairs,encoder1, attn_decoder1, n_iters, print_every=50)
