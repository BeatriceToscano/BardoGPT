import os

import torch
import torch.nn as nn
import math

from DataLoaders.DataLoader_transformer_standard import DataLoader


class MusicEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_length=2048):
        super(MusicEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

        # Inizializzazione dell'embedding posizionale
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        self.register_buffer('pe', torch.zeros(1, max_seq_length, embedding_dim))
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        token_embeddings = self.token_embedding(x)

        positional_embeddings = self.pe[:, :x.size(1), :].to(x.device)

        embeddings = token_embeddings + positional_embeddings
        return embeddings


"""
import json

# Caricamento dei dizionari event_to_idx e idx_to_event
with open('event_to_idx.json', 'r') as f:
    event_to_idx = json.load(f)

with open('idx_to_event.json', 'r') as f:
    idx_to_event = json.load(f)
vocab_size = len(event_to_idx)
dl = DataLoader()
embedding_dim = 512
max_seq_length = 2048
# Supponiamo di avere una sequenza di eventi musicali
example_sequence = dl[0][1]
# Converti la sequenza di eventi nei corrispondenti indici
indexed_sequence = [event_to_idx[event] for event in example_sequence]
music_embedding = MusicEmbedding(vocab_size, embedding_dim, max_seq_length)

# Converti la sequenza di indici in un tensore Torch e aggiungi una dimensione batch
input_indices = torch.tensor(indexed_sequence).unsqueeze(0)  # dimensione: [1, lunghezza_sequenza]

# Ottieni gli embeddings
embeddings = music_embedding(input_indices)

print(embeddings.shape)  # Dovrebbe corrispondere a [1, lunghezza_sequenza, embedding_dim]
"""
