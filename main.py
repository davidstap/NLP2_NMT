from __future__ import unicode_literals, print_function, division
from utils import load_data, Lang, make_bpe, embedding_similarity
from models import AttnDecoderRNN, EncoderLSTM,EncoderGRU,EncoderPositional
from models import EncoderPositional_AIAYN
from eval import evaluate, evaluateFile, evaluateRandomly, bleu_corpus, evaluateAndShowAttention
from training import trainIters
from io import open
import unicodedata
import string
import re
import os
import random

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
max_sent_len = 100
load_pretrained = True
it = str(190000)    #if loading pretrained, choose iteration to load
train = True
train_iters = 110000
lr=0.001
architecture = 'positional'

## Create Byte Pair Encodings
if create_bpe:
    make_bpe()

## Load corpus
input_lang, output_lang, pairs = load_data('train')

## Define encoder and decoder
if architecture == 'AIAYN':
    hidden_size = word_embed_size + pos_embed_size
    encoder = EncoderPositional_AIAYN(input_lang.n_words, word_embed_size, pos_embed_size, max_sent_len)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_sent_len, dropout_p=0.1)

elif architecture == 'positional':
    hidden_size = word_embed_size + pos_embed_size
    encoder = EncoderPositional(input_lang.n_words, word_embed_size, pos_embed_size, max_sent_len)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_sent_len, dropout_p=0.1)

elif architecture == 'LSTM':
    hidden_size = word_embed_size
    encoder = EncoderLSTM(input_lang.n_words, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_sent_len, dropout_p=0.1)

elif architecture == 'GRU':
    hidden_size = word_embed_size
    encoder = EncoderGRU(input_lang.n_words, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_sent_len, dropout_p=0.1)

## Load pre-trained encoder and decoder
if load_pretrained:
    print('Loading models...')
    # encoder.load_state_dict(torch.load('trained_models/GRU/encodergru_it{}'.format(it)))
    # decoder.load_state_dict(torch.load('trained_models/GRU/decoder_it{}'.format(it)))
    encoder.load_state_dict(torch.load('trained_models/EncoderPositional1/encoder_it{}'.format(it)))
    decoder.load_state_dict(torch.load('trained_models/EncoderPositional1/decoder_it{}'.format(it)))
    # encoder.load_state_dict(torch.load('trained_models/positional/encoderpositional_it{}'.format(it)))
    # decoder.load_state_dict(torch.load('trained_models/positional/decoder_it{}'.format(it)))

## Train an encoder/decoder from scratch
if train:
    print('Training model...')
    trainIters(input_lang, output_lang, pairs, encoder, decoder, train_iters,max_sent_len, learning_rate=lr, print_every=1000, plot_every=10)




# input_sentence='an older man is skydiving'
# evaluateAndShowAttention(input_sentence, encoder, decoder, input_lang, output_lang, max_sent_len, device)

quit()
## Make predictions
print('Making predictions...')
savefn='test_preds_{}_it_{}.txt'.format(architecture, it)
evaluateFile(encoder, decoder, input_lang, output_lang, 'test', savefn, max_sent_len, device)

## Calculate evaluation scores
print('Calculating eval scores...')
bleu = bleu_corpus('test', 'predictions/{}'.format(savefn))
print('BLEU score: ',bleu)
os.system("perl multi-bleu.pl -lc data/test/test_2017_flickr.fr < predictions/{}".format(savefn))

## Evaluate n sentences using pre-trained encoder/decoder
# n = 10
# evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, max_sent_len, device,n=n)
# w1 = 'dog.'
# w2 = 'dog'
# print(embedding_similarity(w1, w2, input_lang, encoder))


## List of things to be done
#TODO: initialize hidden states with word embedding averages instead of zeros
#TODO: make predictions
#TODO: calculate several scores
#TODO: fix fucking Java TER
#TODO: add METEOR code
#TODO: worden word embeddings geupdatet?
#TODO: linear neural layer ipv
