from __future__ import unicode_literals, print_function, division
from utils import load_data, Lang, MAX_LENGTH, make_bpe
from models import AttnDecoderRNN, EncoderRNN, DecoderRNN,EncoderPositional, EncoderPositionalSimple
from training import trainIters
from io import open
import unicodedata
import string
import re
import random
from eval import bleu_corpus

# Pytorch imports
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TODO: fix input (small letters etc.)
#TODO: worden word embeddings geupdatet?
#TODO: linear neural layer ipv

create_bpe = False

if create_bpe:
    make_bpe()





# Load corpus
input_lang, output_lang, pairs = load_data('train')

# Define encoder/decoder model
word_embed_size = 256
pos_embed_size = 20
hidden_size = word_embed_size + pos_embed_size
max_length = 50

# encoder = EncoderPositionalSimple(input_lang.n_words, hidden_size).to(device)
# encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
encoder = EncoderPositional(input_lang.n_words, word_embed_size, pos_embed_size).to(device)
# encoder = EncoderPositionalSimple(input_lang.n_words, word_embed_size).to(device)


attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

for param in encoder.parameters():
  print(param.data)


# Train model
# n_iters = 75000
n_iters = 110

trainIters(input_lang, output_lang, pairs, encoder, attn_decoder, n_iters, print_every=500)
print('finished training')

for param in encoder.parameters():
  print(param.data)



#TODO make predictions


### Evaluation
# bleu_corpus(references, predictions)
