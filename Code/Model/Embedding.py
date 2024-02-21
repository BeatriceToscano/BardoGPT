import math
import torch
import torch.nn as nn

from DataLoaders.pretty_midi_DataLoader import DataLoader


class MusicEmbedding(nn.Module):
    def __init__(self, embedding_dim=512, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(MusicEmbedding, self).__init__()
        self.device = device
        self.pitch_embedding = nn.EmbeddingBag(128, embedding_dim, mode='sum', device=self.device).to(self.device)
        self.velocity_embedding = nn.EmbeddingBag(128, embedding_dim, mode='sum', device=self.device).to(self.device)
        self.program_embedding = nn.EmbeddingBag(128, embedding_dim, mode='sum', device=self.device).to(self.device)
        self.drum_embedding = nn.EmbeddingBag(2, embedding_dim, mode='sum', device=self.device).to(self.device)
        self.embedding_dim = embedding_dim
        self.to(self.device)

    def temporal_positional_encoding(self, continuous_features):
        # Assumi che continuous_features abbia dimensione [batch_size, num_features], dove num_features è 2 per start e end
        # Calcola le componenti seno e coseno usando il tempo
        time = continuous_features.to(self.device).unsqueeze(-1) * 10000 ** (torch.arange(0, self.embedding_dim, 2, device=self.device).float() / self.embedding_dim)
        sin_component = torch.sin(time)
        cos_component = torch.cos(time)
        time_embeddings = torch.cat((sin_component, cos_component), dim=-1)
        return time_embeddings

    def forward(self, pitch_indices, velocity_indices, program_indices, continuous_features, drum_indices):
        # Non sono necessari gli offsets se ogni chiamata processa un singolo esempio
        pitch_embeddings = self.pitch_embedding(pitch_indices.to(self.device).unsqueeze(0)).to(self.device)
        velocity_embeddings = self.velocity_embedding(velocity_indices.to(self.device).unsqueeze(0)).to(self.device)
        program_embeddings = self.program_embedding(program_indices.to(self.device).unsqueeze(0)).to(self.device)
        drum_embeddings = self.drum_embedding(drum_indices.to(self.device).unsqueeze(0)).to(self.device)

        # Genera la codifica posizionale temporale
        time_embeddings = self.temporal_positional_encoding(continuous_features).to(self.device)

        # Concatena gli embeddings
        combined_embeddings = torch.cat([drum_embeddings,
                                         time_embeddings[0].reshape(1, -1),
                                         time_embeddings[1].reshape(1, -1),
                                         program_embeddings,
                                         pitch_embeddings,
                                         velocity_embeddings
                                         ], dim=-1).to(self.device)

        return combined_embeddings
