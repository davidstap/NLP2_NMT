from nltk import word_tokenize

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
    french = preprocess(load_file('data{}'.format(fn_f)), preprocess_type)

    return [english, french]

def load_file(fn):
    with open(fn, 'r') as f:
        return [x.split() for x in f.read().splitlines()]

def preprocess(data, type):
    if type == 'lowercasing':
        return [[w.lower() for w in s] for s in data]
