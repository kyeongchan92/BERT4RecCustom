import math

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

    def __init__(self, vocab_size, embed_size, max_len, attrs_emb_sizes, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        if attrs_emb_sizes:
            self.attrs = []
            for attr_size in attrs_emb_sizes.values():
                self.attrs.append(nn.Embedding(attr_size, embed_size))
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

        self.printer = PrintInputShape(2)

    def forward(self, sequence, attrs_idxs):
        # print(self.printer.cnt)
        self.printer.print(sequence, notation='sequence')
        self.printer.print(attrs_idxs, notation='attrs_idxs')
        
        attr_emb_sum = torch.empty(len(sequence), self.embed_size)
        for idxs, attr_emb_mat in zip(attrs_idxs, self.attrs):
            attr_emb_sum += attr_emb_mat(idxs)

        x = self.token(sequence) + self.position(sequence) + attr_emb_sum
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

