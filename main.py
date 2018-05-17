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

#TODO: fix input (small letters etc.)
#TODO: worden word embeddings geupdatet?
#TODO: linear neural layer ipv

#####################################################
#################### Params #########################
#####################################################
create_bpe = False
# Define encoder/decoder model
word_embed_size = 256
pos_embed_size = 20
hidden_size = word_embed_size + pos_embed_size
max_sent_len = 200


# Load corpus
#TODO: use more data (now only 250 entries, see utils)
input_lang, output_lang, pairs = load_data('train')

# import numpy as np
# test = np.array(pairs)
# print(max([len(x) for x in test[1,:]]))

it=str(200000)



# encoder = torch.load()
# decoder = torch.load()



# encoder.load_state_dict(torch.load('trained_models/encoder_it{}'.format(it)))
# encoder.load_state_dict(torch.load('trained_models/decoder_it{}'.format(it)))
encoder = EncoderPositional(input_lang.n_words, word_embed_size, pos_embed_size,max_sent_len).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_sent_len,dropout_p=0.1).to(device)

encoder.load_state_dict(torch.load('trained_models/encoder_it{}'.format(it)))
decoder.load_state_dict(torch.load('trained_models/decoder_it{}'.format(it)))

print(encoder)
print(decoder)

encoder.hidden_size=word_embed_size + pos_embed_size
decoder.hidden_size=word_embed_size + pos_embed_size



evaluateRandomly(encoder, decoder, input_lang,output_lang, pairs, max_sent_len, device)





quit()




if create_bpe:
    make_bpe()

# encoder = EncoderPositionalSimple(input_lang.n_words, hidden_size).to(device)
# encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
encoder = EncoderPositional(input_lang.n_words, word_embed_size, pos_embed_size,max_sent_len).to(device)
# encoder = EncoderPositionalSimple(input_lang.n_words, word_embed_size).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_sent_len,dropout_p=0.1).to(device)

# evaluateRandomly(encoder,attn_decoder, input_lang, output_lang, pairs, max_sent_len, device)

# compare sum sum before and after training
# for param in encoder.parameters():
#     print(sum(sum(param.data)))

# Train model
# n_iters = 75000
n_iters = 2900000

trainIters(input_lang, output_lang, pairs, encoder, attn_decoder, n_iters,max_sent_len, print_every=2500)
print('finished training')

# compare sum sum before and after training
# for param in encoder.parameters():
#   print(sum(sum(param.data)))

# Findings:
# - 'Positional embeddings' are trained (i.e. weights change)
# - 'Word embeddings' are trained (i.e. weights change)

#####################################################
#################### Predictions ####################
#####################################################
#TODO: make predictions

#####################################################
#################### Evaluation #####################
#####################################################
#TODO: calculate several scores
#TODO: fix fucking Java TER
#TODO: add METEOR code
# bleu_corpus(references, predictions)
