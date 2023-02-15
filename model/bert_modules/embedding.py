import math

import torch
import torch.nn as nn

from utils import PrintInputShape

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, use_attributes, attrs_each_size, i2attr_map, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.use_attributes = use_attributes
        self.printer = PrintInputShape(2)

        if use_attributes:
            self.i2attr_map = i2attr_map
            self.attrs_emb_mats = {}
            for attr_name, size in attrs_each_size.items():
                self.attrs_emb_mats[attr_name] = nn.Embedding(size, embed_size)

    def forward(self, sequence):
        # print(self.printer.cnt)
        # self.printer.print(sequence, notation='sequence')
        # self.printer.print(attrs_idxs, notation='attrs_idxs')
        if self.use_attributes:
            x, idxs = sequence
            x = self.token(sequence) + self.position(sequence)
            attr_sum = torch.empty(len(sequence), self.embed_size)
            for attr_name in self.i2attr_map:
                self.printer.print(sequence, 'sequence')
                idxs = torch.LongTensor([self.i2attr_map[attr_name][item] for item in sequence])
                attr_sum += self.attrs_emb_mats[attr_name](idxs)
            x += attr_sum
        else:
            x = self.token(sequence) + self.position(sequence)
        
        return self.dropout(x)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

