from __future__ import unicode_literals, print_function, division
from utils import load_data, Lang, make_bpe, embedding_similarity
from models import AttnDecoderRNN, EncoderLSTM,EncoderGRU,EncoderPositional,CustomDecoderRNN
from training import trainIters
from io import open
import unicodedata
import string
import re
import random
from eval import evaluate, evaluateRandomly, bleu_corpus

# Pytorch imports
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Set hyperparamters
create_bpe = False
word_embed_size = 256
pos_embed_size = 20
hidden_size = word_embed_size + pos_embed_size
# hidden_size = 256
max_sent_len = 50
load_pretrained = False

## Load corpus
if create_bpe:
    make_bpe()
input_lang, output_lang, pairs = load_data('train')

## Define encoder and decoder
encoder = EncoderPositional(input_lang.n_words, word_embed_size, pos_embed_size, max_sent_len)
# encoder = EncoderGRU(input_lang.n_words, word_embed_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_sent_len, dropout_p=0.1)

## Load pre-trained encoder and decoder
if load_pretrained:
    it=str(580000)
    encoder.load_state_dict(torch.load('trained_models/encoder_it{}'.format(it)))
    decoder.load_state_dict(torch.load('trained_models/decoder_it{}'.format(it)))

## Train an encoder/decoder from scratch
else:
    print('Training model...')
    n_iters = 100
    trainIters(input_lang, output_lang, pairs, encoder, decoder, n_iters,max_sent_len, print_every=50)

## Evaluate n sentences using pre-trained encoder/decoder
n = 10
# evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, max_sent_len, device,n=n)
w1 = 'dog.'
w2 = 'dog'
print(embedding_similarity(w1, w2, input_lang, encoder))


## List of things to be done
#TODO: initialize hidden states with word embedding averages instead of zeros
#TODO: make predictions
#TODO: calculate several scores
#TODO: fix fucking Java TER
#TODO: add METEOR code
#TODO: fix input (small letters etc.)
#TODO: worden word embeddings geupdatet?
#TODO: linear neural layer ipv
# bleu_corpus(references, predictions)
