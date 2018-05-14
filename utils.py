from nltk import word_tokenize
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 100

def load_data(data_type, preprocess_type):
    if data_type == 'train':
        fn_e = '/train/train.en'
        fn_f = '/train/train.fr'
    elif data_type == 'val':
        fn_e = '/val/val.en'
        fn_f = '/val/val.fr'
    elif data_type == 'test':
        fn_e = '/test/test_2017_flickr.en'
        fn_f = '/test/test_2017_flickr.fr'

    english = preprocess(load_file('data{}'.format(fn_e)), preprocess_type)
    input_lang = Lang(english)
    french = preprocess(load_file('data{}'.format(fn_f)), preprocess_type)
    output_lang = Lang(french)

    pairs = list(zip(english, french))
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs

def load_file(fn):
    with open(fn, 'r') as f:
        return [x.split() for x in f.read().splitlines()]

def preprocess(data, type):
    if type == 'lowercasing':
        return [[w.lower() for w in s] for s in data]

# Class to keep track of word indices and counts
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
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
    return [lang.word2index[word] for word in sentence]

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
