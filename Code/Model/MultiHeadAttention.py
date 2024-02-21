import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size=512*6, heads=64, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.dropout = nn.Dropout(0.1)

        self.device = device
        self.to(self.device)

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False).to(self.device)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False).to(self.device)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False).to(self.device)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size).to(self.device)

    def forward(self, values, keys, queries, mask=False):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim).to(self.device)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim).to(self.device)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim).to(self.device)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Calcolo dell'attenzione
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask:
            mask = torch.triu(torch.ones_like(attention ), diagonal=1).bool()
            attention = attention.masked_fill(torch.Tensor(mask == 0), float("-1e20"))

        attention = torch.softmax(attention / (self.head_dim ** (1 / 2)), dim=3)
        attention = self.dropout(attention)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out
