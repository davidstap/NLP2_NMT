import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

# Positional Encoder that outputs a 'sentence matrix' containing all word
# embeddings of the sentence concatenated with embeddings of the position of
# the word.
class EncoderPositional(nn.Module):
    def __init__(self, input_size, word_embed_size, pos_embed_size, max_length):
        super(EncoderPositional,self).__init__()
        self.hidden_size = word_embed_size + pos_embed_size
        self.max_length = max_length
        self.word_embedding = nn.Embedding(input_size, word_embed_size)
        self.pos_embedding = nn.Embedding(self.max_length, pos_embed_size)
        self.embed_combine = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, inputs):
        # Transform inputs to embedding matrix
        output = torch.zeros(self.max_length, self.hidden_size, device=device)
        for i, input in enumerate(inputs):
            word_embedding = self.word_embedding(input).view(-1)
            pos_embedding = self.pos_embedding(torch.tensor(i)).view(-1)
            c = torch.cat((word_embedding,pos_embedding))
            output[i] = F.sigmoid(self.embed_combine(c))

        # Compute hidden state as average over word embeddings
        hidden = output[0:len(inputs),:].mean(dim=0).view(1,1,-1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def AIAYN_attention_init(max_length, pos_embed_size):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    attention_init = np.array([
        [pos / np.power(10000, 2*i/pos_embed_size) for i in range(pos_embed_size)]
        if pos != 0 else np.zeros(pos_embed_size) for pos in range(max_length)])

    attention_init[1:, 0::2] = np.sin(attention_init[1:, 0::2]) # dim 2i
    attention_init[1:, 1::2] = np.cos(attention_init[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(attention_init).type(torch.FloatTensor)

# Positional Encoder that outputs a 'sentence matrix' containing all word
# embeddings of the sentence concatenated with embeddings of the position of
# the word.
class EncoderPositional_AIAYN(nn.Module):
    def __init__(self, input_size, word_embed_size, pos_embed_size, max_length):
        super(EncoderPositional_AIAYN, self).__init__()
        self.hidden_size = word_embed_size + pos_embed_size
        self.max_length = max_length
        self.word_embedding = nn.Embedding(input_size, word_embed_size)
        self.pos_embedding = nn.Embedding(self.max_length, pos_embed_size)
        self.embed_combine = nn.Linear(self.hidden_size, self.hidden_size)

        # Initialize sinuslike (as in AIAYN)
        self.pos_embedding.weight.data = AIAYN_attention_init(self.max_length, pos_embed_size)

    def forward(self, inputs):
        # Transform inputs to embedding matrix
        output = torch.zeros(self.max_length, self.hidden_size, device=device)
        for i, input in enumerate(inputs):
            word_embedding = self.word_embedding(input).view(-1)
            pos_embedding = self.pos_embedding(torch.tensor(i)).view(-1)
            c = torch.cat((word_embedding,pos_embedding))
            output[i] = F.sigmoid(self.embed_combine(c))

        # Compute hidden state as average over word embeddings
        hidden = output[0:len(inputs),:].mean(dim=0).view(1,1,-1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    # def __init__(self, input_size, word_embed_size, max_length):
    #     super(EncoderPositional_AIAYN,self).__init__()
    #     self.hidden_size = word_embed_size
    #     self.max_length = max_length
    #     self.word_embedding = nn.Embedding(input_size, word_embed_size)
    #
    #     # Create an additional embedding layer for positions
    #     # self.pos_embedding = nn.Embedding(self.max_length, pos_embed_size)
    #     self.pos_embedding = nn.Embedding(self.max_length, word_embed_size)
    #     self.pos_embedding.weight.data = AIAYN_attention_init(self.max_length, word_embed_size)
    #
    #     self.embed_combine = nn.Linear(self.hidden_size, self.hidden_size)
    # def forward(self, inputs):
    #     # Transform inputs to embedding matrix
    #     output = torch.zeros(self.max_length, self.hidden_size, device=device)
    #     for i, input in enumerate(inputs):
    #         word_embedding = self.word_embedding(input).view(-1)
    #         pos_embedding = self.pos_embedding(torch.tensor(i)).view(-1)
    #
    #         # output[i] = word_embedding + pos_embedding
    #         output[i] = F.sigmoid(self.embed_combine(word_embedding + pos_embedding))
    #
    #     # Compute hidden state as average over word embeddings
    #     hidden = output[0:len(inputs),:].mean(dim=0).view(1,1,-1)
    #     return output, hidden
    #
    # def initHidden(self):
    #     return torch.zeros(1, 1, self.hidden_size, device=device)



# Positional Encoder that outputs a 'sentence matrix' containing all word
# embeddings of the sentence concatenated with a simple word index (float)
# in the sentence of the corresponding word. Word embedding size is reduced
# by 1 in order to let total positional embedding size be hidden_size
class EncoderPositionalSimple(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderPositional,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size-1)

    def forward(self, inputs, max_length):
        # Transform inputs to embedding matrix
        output = torch.zeros(max_length, self.hidden_size, device=device)
        for i, input in enumerate(inputs):
            word_embedding = self.embedding(input).view(-1)

            # Add word position to create positional embedding
            position = torch.tensor([i], dtype=torch.float)
            output[i] = torch.cat((word_embedding,position))

        # Compute hidden state as average over word embeddings
        hidden = output[0:len(inputs),:].mean(dim=0).view(1,1,-1)
        return output, hidden

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device), \
            torch.zeros(1, 1, self.hidden_size, device=device))


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class CustomDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, bilinear=False,dropout_p=None):
        super(CustomDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.bilinear = bilinear

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, src_len):
        embedded = self.embedding(input).view(1, 1, -1)

        # Compute hidden state using 'standard' decoder
        output, hidden = self.gru(embedded, hidden)

        # Compute scores by dot product/bilinear of decoder hidden state and encoder hidden states
        if self.bilinear == False:
            # Scoring function is simple dot product
            scores = torch.mm(hidden.squeeze(0), torch.transpose(encoder_outputs,0,1))
        else:
            # TODO implement bilinear scoring function (called general in paper)
            scores = None

        # Normalize to obtain weights for weighted average over encoder's hidden states
        attn_weights = torch.zeros(1,encoder_outputs.shape[0])

        # Perform softmax over the encoder's actual hidden states only instead of over all
        # (partly non existent, all zeros) states
        attn_weights[0,0:src_len] = F.softmax(scores[0,0:src_len],dim=0)

        # Get weighted avr of encoder hidden states using attn_weights (alpha)
        c = torch.bmm(attn_weights.unsqueeze(0),
                    encoder_outputs.unsqueeze(0))

        # Concat c with current decoder hidden state
        conc = torch.cat((hidden,c), dim=2)

        # Feed concatenation through linear layer to obtain 'final' decoder hidden state
        hidden = self.attn_combine(conc)

        # Compute distribution over target vocabulary using hidden state
        output = F.log_softmax(self.out(hidden[0]), dim=1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
# class AttnDecoderRNN_AIAYN(nn.Module):
#     def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
#         super(AttnDecoderRNN_AIAYN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         self.word_embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.pos_embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, pos, hidden, encoder_outputs):
#
#
#         # include positional embedding
#         embedded = self.word_embedding(input).view(1, 1, -1)
#         embedded += self.pos_embedding(pos).view(1,1,-1)
#         embedded = self.dropout(embedded)
#
#
#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))
#
#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)
#
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#
#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
