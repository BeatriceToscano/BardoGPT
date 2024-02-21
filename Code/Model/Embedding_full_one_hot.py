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
        self.tempo_embedding = nn.EmbeddingBag(8, embedding_dim, mode='sum', device=self.device).to(self.device)
        self.drum_embedding = nn.EmbeddingBag(2, embedding_dim, mode='sum', device=self.device).to(self.device)
        self.embedding_dim = embedding_dim
        self.to(self.device)

    def forward(self, pitch_indices, velocity_indices, program_indices, tempo_indices, drum_indices):
        # Non sono necessari gli offsets se ogni chiamata processa un singolo esempio
        pitch_embeddings = self.pitch_embedding(pitch_indices.to(self.device).unsqueeze(0)).to(self.device)
        velocity_embeddings = self.velocity_embedding(velocity_indices.to(self.device).unsqueeze(0)).to(self.device)
        tempo_embeddings = self.program_embedding(tempo_indices.to(self.device).unsqueeze(0)).to(self.device)
        program_embeddings = self.program_embedding(program_indices.to(self.device).unsqueeze(0)).to(self.device)
        drum_embeddings = self.drum_embedding(drum_indices.to(self.device).unsqueeze(0)).to(self.device)

        # Concatena gli embeddings
        combined_embeddings = torch.cat([drum_embeddings,
                                         tempo_embeddings,
                                         program_embeddings,
                                         pitch_embeddings,
                                         velocity_embeddings
                                         ], dim=-1).to(self.device)

        return combined_embeddings
