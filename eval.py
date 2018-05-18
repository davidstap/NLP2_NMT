import torch
import nltk
import random
from utils import tensorFromSentence, cmdline
import os
from utils import load_data, load_file, reverseBPE

def evaluate(encoder, decoder, sentence, input_lang,output_lang, max_length, device):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)


        # for ei in range(input_length):
        #     # encoder_output, encoder_hidden = encoder(input_tensor[ei],
        #                                              # encoder_hidden)
        #     encoder_output, encoder_hidden = encoder(input_tensor[ei], max_length)
        #     encoder_outputs[ei] += encoder_output[0, 0]

        encoder_outputs, encoder_hidden = encoder(input_tensor, max_length)

        decoder_input = torch.tensor([[0]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == 1: #EOS token
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, input_lang,output_lang, pairs, max_length, device, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        input_sentence = ' '.join(w for w in pair[0])

        print('> ', reverseBPE(input_sentence))
        trans_sentence = ' '.join(w for w in pair[1])
        print('< ', reverseBPE(trans_sentence))

        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang,max_length, device)
        output_sentence = ' '.join(output_words)

        print('< ', reverseBPE(output_sentence))
        print('')

def evaluateFile(encoder, decoder, input_lang, output_lang, datatype, savefn, max_length, device):
    _, _, pairs = load_data(datatype)

    f = open(savefn,'wb')
    
    for pair in pairs:
        input_sentence = reverseBPE(' '.join(w for w in pair[0]))

        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang, max_length, device)
        predi_sentence = reverseBPE(' '.join(output_words))

        f.write(predi_sentence)

    f.close()


def bleu_corpus(referencefn, hypothesisfn):
    _,_,ref = load_data(referencefn)

    # fix correct format for Bleu reference
    references =[[reverseBPE(' '.join(s[1])).split()] for s in ref]
    hypothesis = load_file(hypothesisfn)

    return nltk.translate.bleu_score.corpus_bleu(references, hypothesis)
