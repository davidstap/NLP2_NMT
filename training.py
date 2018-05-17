import random
import time
import math
from utils import tensorsFromPair

import torch
from torch import optim
import torch.nn as nn

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trainIters(input_lang,output_lang,pairs, encoder, decoder, n_iters, max_length, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Create an parameter optimization object for both models
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # Pick a random sentence and convert to tensor of word indices
    training_pairs = [tensorsFromPair(input_lang,output_lang,random.choice(pairs))
                      for i in range(n_iters)]

    # Use negative log likelihood as loss
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]

        # Enforce max sentence length
        input_tensor = training_pair[0]
        if len(input_tensor) > max_length:
            continue
        target_tensor = training_pair[1]

        # Train model using one sentence pair, returns the negative log likelihood
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if iter % 10000 == 0:
            torch.save(encoder.state_dict(), 'trained_models/encoder_it{}'.format(iter))
            torch.save(decoder.state_dict(), 'trained_models/decoder_it{}'.format(iter))

    showPlot(plot_losses)



def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, \
    decoder_optimizer, criterion, max_length, teacher_forcing_ratio = 0.5):
    # Init encoder and set gradients to 0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    # Init encoder output size
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Loop over input words and compute intermediate outputs and hidden states
    if 'Positional' in type(encoder).__name__:
        encoder_outputs, encoder_hidden = encoder(input_tensor)
    else:
        # Tuple containing the hidden AND cell state in case of LSTM
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # Loop over input words and compute intermediate outputs and hidden states
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

    # Init decoder
    decoder_input = torch.tensor([[0]], device=device)

    # In case of LSTM, unpack hidden state from tuple
    if isinstance(encoder_hidden,tuple):
        encoder_hidden = encoder_hidden[0]

    # Initialize decoder hidden state with encoder's context vector
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, input_length)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs,input_length)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == 1:
                break

    # Compute loss and gradients, update weights
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '{}m {}s'.format(m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    try:
        es = s / (percent)
    except ZeroDivisionError:
        es = 0
    rs = es - s
    return '{} (- {})'.format(asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
