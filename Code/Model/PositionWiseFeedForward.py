import torch
import torch.nn as nn


class PositionWiseFeedforward(nn.Module):
    def __init__(self, embed_size=6*512, ff_dim=12*512, dropout_rate=0.1, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(PositionWiseFeedforward, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(embed_size, ff_dim).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.fc2 = nn.Linear(ff_dim, embed_size).to(self.device)
        self.dropout = nn.Dropout(dropout_rate).to(self.device)
        self.to(self.device)

    def forward(self, x):
        out = self.fc1(x).to(self.device)
        out = self.relu(out).to(self.device)
        out = self.dropout(out).to(self.device)
        out = self.fc2(out).to(self.device)
        return out
