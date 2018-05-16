import nltk

def bleu_corpus(references, hypothesis):
    return nltk.translate.bleu_score.corpus_bleu(references, hypothesis)

# how to use    
# references = [[['It', 'is', 'a', 'cat', 'at', 'room']]]
# hypothesis = [['It', 'is', 'a', 'cat', 'at', 'room']]
# #there may be several references
#
# bleu_corpus(references, hypothesis)
