import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter, init
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class SlotRelatedModule(nn.Module):
    def __init__(self, max_sentence_length, num_of_slot):

        super(SlotRelatedModule, self).__init__()

        self.__output_dim = 2 * num_of_slot
        self.embedding = nn.Embedding(num_embeddings=max_sentence_length,
                                      embedding_dim=self.__args.word_embedding_dim)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__args.__embedding_dim,
            hidden_size=self.__args.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1
        )
        self.__linear_layer = nn.Linear(
            self.__args.__hidden_dim,
            self.__output_dim)
        self.sigmoid_layer = nn.Sigmoid()

        self.turn = 0

    def forward(self, text_embedding, seq_lens):  # B, sentence length, embedding_dim
        packed_text = pack_padded_sequence(text_embedding, seq_lens, batch_first=True)
        lstm_hidden = self.__lstm_layer(packed_text)
        result = self.__linear_layer(lstm_hidden)
        result = self.sigmoid_layer(result)
        padded_hidden, _ = pad_packed_sequence(lstm_hidden, batch_first=True)
        padded_result = pad_packed_sequence(result, batch_first=True)

        return padded_result, torch.cat([padded_hidden[i][:seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0)
