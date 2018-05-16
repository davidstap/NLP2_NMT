import torch
import nltk
import random
from utils import tensorFromSentence

def evaluate(encoder, decoder, sentence, input_lang, max_length, device):
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


        print(decoder_input.size())
        print(decoder_hidden.size())
        print(encoder_outputs.size())



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

def evaluateRandomly(encoder, decoder, input_lang, pairs, max_length, device, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, max_length, device)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def bleu_corpus(references, hypothesis):
    return nltk.translate.bleu_score.corpus_bleu(references, hypothesis)

# how to use bleu_corpus:
# references = [[['It', 'is', 'a', 'cat', 'at', 'room']]]
# hypothesis = [['It', 'is', 'a', 'cat', 'at', 'room']]
# #there may be several references
#
# bleu_corpus(references, hypothesis)
