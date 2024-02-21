import torch
from torch import nn

from Model.AddNorm import AddNorm
from Model.MultiHeadAttention import MultiHeadAttention
from Model.PositionWiseFeedForward import PositionWiseFeedforward


class Encoder(nn.Module):
    def __init__(self, embed_size=512, num_params=6, heads=64, ff_dim=12 * 512, dropout_rate=0.1, eps=1e-6, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(Encoder, self).__init__()
        self.device = device
        self.attention = MultiHeadAttention(embed_size * num_params, heads, device=self.device).to(self.device)
        self.add_norm_1 = AddNorm(embed_size * num_params, eps, device=self.device).to(self.device)
        self.feed_forward = PositionWiseFeedforward(embed_size * num_params, ff_dim, dropout_rate, device=self.device).to(self.device)
        self.add_norm_2 = AddNorm(embed_size * num_params, eps, device=self.device).to(self.device)
        self.to(self.device)

    def forward(self, combined_embeddings):
        mha = self.attention(combined_embeddings, combined_embeddings, combined_embeddings).to(self.device)
        an = self.add_norm_1(combined_embeddings, mha).to(self.device)
        output = self.feed_forward(an).to(self.device)
        return self.add_norm_2(an, output)
