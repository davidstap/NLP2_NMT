from utils import MAX_LENGTH
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Positional Encoder that outputs a 'sentence matrix' containing all word
# embeddings of the sentence concatenated with embeddings of the position of
# the word.
class EncoderPositional(nn.Module):
    def __init__(self, input_size, word_embed_size, pos_embed_size, max_length = MAX_LENGTH):
        super(EncoderPositional,self).__init__()
        self.hidden_size = word_embed_size + pos_embed_size
        self.word_embedding = nn.Embedding(input_size, word_embed_size)

        # Create an additional embedding layer for positions
        self.pos_embedding = nn.Embedding(max_length, pos_embed_size)

    def forward(self, inputs, max_length = MAX_LENGTH):
        # Transform inputs to embedding matrix
        output = torch.zeros(max_length, self.hidden_size, device=device)
        for i, input in enumerate(inputs):
            word_embedding = self.word_embedding(input).view(-1)
            pos_embedding = self.pos_embedding(torch.tensor(i)).view(-1)

            # Concat word and position embedding to create positional embedding
            # TODO nn.linear ipv cat
            output[i] = torch.cat((word_embedding,pos_embedding))

        # Compute hidden state as average over word embeddings
        hidden = output[0:len(inputs),:].mean(dim=0).view(1,1,-1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Positional Encoder that outputs a 'sentence matrix' containing all word
# embeddings of the sentence concatenated with a simple word index (float)
# in the sentence of the corresponding word. Word embedding size is reduced
# by 1 in order to let total positional embedding size be hidden_size
class EncoderPositionalSimple(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderPositional,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size-1)

    def forward(self, inputs, max_length = MAX_LENGTH):
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

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
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
