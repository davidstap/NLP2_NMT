from __future__ import unicode_literals, print_function, division
from utils import load_data, Lang, make_bpe
from models import AttnDecoderRNN, EncoderRNN, DecoderRNN,EncoderPositional, EncoderPositionalSimple
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
max_sent_len = 200
load_pretrained = True

## Load corpus
if create_bpe:
    make_bpe()
input_lang, output_lang, pairs = load_data('train')
print(input_lang.word2index)

## Define encoder and decoder
encoder = EncoderPositional(input_lang.n_words, word_embed_size, pos_embed_size, max_sent_len).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_sent_len,dropout_p=0.1).to(device)
# encoder = EncoderPositionalSimple(input_lang.n_words, hidden_size).to(device)
# encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)

## Load pre-trained encoder and decoder
if load_pretrained:
    it=str(580000)
    encoder.load_state_dict(torch.load('trained_models/encoder_it{}'.format(it)))
    decoder.load_state_dict(torch.load('trained_models/decoder_it{}'.format(it)))

    print(encoder)
    print(decoder)
    encoder.hidden_size=word_embed_size + pos_embed_size
    decoder.hidden_size=word_embed_size + pos_embed_size

## Train an encoder/decoder from scratch
else:
    n_iters = 2900000
    trainIters(input_lang, output_lang, pairs, encoder, attn_decoder, n_iters,max_sent_len, print_every=2500)

## Evaluate n sentences using pre-trained encoder/decoder
n = 10
evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, max_sent_len, device,n=n)

# compare sum sum before and after training
# for param in encoder.parameters():
#     print(sum(sum(param.data)))

#TODO: make predictions
#TODO: calculate several scores
#TODO: fix fucking Java TER
#TODO: add METEOR code
#TODO: fix input (small letters etc.)
#TODO: worden word embeddings geupdatet?
#TODO: linear neural layer ipv
# bleu_corpus(references, predictions)
