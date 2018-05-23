from nltk import word_tokenize
import torch
import torch.nn.functional as F
from itertools import chain
from glob import glob
import os
from subprocess import PIPE, Popen
from nltk import RegexpTokenizer
import string

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(data_type):
    if data_type == 'train':
        fn_e = '/train/train.en.BPE'
        fn_f = '/train/train.fr.BPE'
    elif data_type == 'val':
        fn_e = '/val/val.en.BPE'
        fn_f = '/val/val.fr.BPE'
    elif data_type == 'test':
        fn_e = '/test/test_2017_flickr.en.BPE'
        fn_f = '/test/test_2017_flickr.fr.BPE'

    english = load_file('data{}'.format(fn_e))
    input_lang = Lang(english)
    french = load_file('data{}'.format(fn_f))
    output_lang = Lang(french)

    pairs = list(zip(english, french))
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs

def load_file(fn):
    with open(fn, 'r') as f:
        return [s.split() for s in f.read().splitlines()]


def make_bpe():
    # create preprocessed files
    preprocess('data/train/train.en', 'data/train/train_lc.en')
    preprocess('data/train/train.fr', 'data/train/train_lc.fr')
    preprocess('data/test/test_2017_flickr.en', 'data/test/test_2017_flickr_lc.en')
    preprocess('data/test/test_2017_flickr.fr', 'data/test/test_2017_flickr_lc.fr')
    preprocess('data/val/val.en', 'data/val/val_lc.en')
    preprocess('data/val/val.fr', 'data/val/val_lc.fr')

    # create BPE vocabs (by combining English and French)
    os.system("python subword-nmt/learn_joint_bpe_and_vocab.py --input data/train/train_lc.en data/train/train_lc.fr -s 10000 -o data/bpe/ef_codes --write-vocabulary data/bpe/vocab_file_en data/bpe/vocab_file_fr")

    # translate original data to BPE representation (e.g. English stored in data/train/train.en.BPE)
    os.system("python subword-nmt/apply_bpe.py -c data/bpe/ef_codes --vocabulary data/bpe/vocab_file_en --vocabulary-threshold 1 < data/train/train_lc.en > data/train/train.en.BPE")
    os.system("python subword-nmt/apply_bpe.py -c data/bpe/ef_codes --vocabulary data/bpe/vocab_file_fr --vocabulary-threshold 1 < data/train/train_lc.fr > data/train/train.fr.BPE")
    os.system("python subword-nmt/apply_bpe.py -c data/bpe/ef_codes --vocabulary data/bpe/vocab_file_en --vocabulary-threshold 1 < data/val/val_lc.en > data/val/val.en.BPE")
    os.system("python subword-nmt/apply_bpe.py -c data/bpe/ef_codes --vocabulary data/bpe/vocab_file_fr --vocabulary-threshold 1 < data/val/val_lc.fr > data/val/val.fr.BPE")
    os.system("python subword-nmt/apply_bpe.py -c data/bpe/ef_codes --vocabulary data/bpe/vocab_file_en --vocabulary-threshold 1 < data/test/test_2017_flickr_lc.en > data/test/test_2017_flickr.en.BPE")
    os.system("python subword-nmt/apply_bpe.py -c data/bpe/ef_codes --vocabulary data/bpe/vocab_file_fr --vocabulary-threshold 1 < data/test/test_2017_flickr_lc.fr > data/test/test_2017_flickr.fr.BPE")


def preprocess(fn, newfn):
    with open(fn) as f:
        sentences = [s for s in f]

    sentences = [RegexpTokenizer(r'''\w'|\w+|[^\w\s]''').tokenize(s.decode('utf-8').lower()) for s in sentences]
    sentences = [s[:-1] if s[-1] == '.' else s for s in sentences]

    f = open(newfn,'wb')
    for s in sentences:
        # f.write(' '.join(w for w in s))
        f.write(' '.join(w for w in s).encode('utf-8')+'\n')
    f.close()

# Class to keep track of word indices and counts
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", -1: "-UNK-"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Return a list of indices for words in a sentence
def indexesFromSentence(lang, sentence):
    #initialize with BOS token
    indexes = [0]
    for w in sentence:
        try:
            indexes.append(lang.word2index[w])
        except KeyError:
            # Only happens once in test set
            #indexes.append(-1)
            continue

    return indexes


# Return a tensor representing a sentence using indices and EOS token
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(1)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# Return tensor pair of sentence pair
def tensorsFromPair(input_lang,output_lang,pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]

def reverseBPE(s):
    return cmdline('echo "{}" | sed -E "s/(@@ )|(@@ ?$)//g"'.format(s.replace("<EOS>", "")))

# Convert word to vector using encoder
def embedding_similarity(w1, w2, lang, encoder):
    w1 = encoder.word_embedding(torch.tensor([lang.word2index[w1]])).view(-1)
    w2 = encoder.word_embedding(torch.tensor([lang.word2index[w2]])).view(-1)
    return F.cosine_similarity(w1,w2,0)

import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
sns.set()

def make_plot(fn):
    # assumption: los every 10 steps
    y = pickle.load(open(fn, 'rb'))
    x = np.arange(10,len(y)*10+1,10)
    plt.plot(x[0::15],y[0::15])
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.show()
