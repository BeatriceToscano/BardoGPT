import torch
import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self, embed_size=6*512, eps=1e-6, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(AddNorm, self).__init__()
        self.device = device
        self.norm = nn.LayerNorm(embed_size, eps=eps).to(self.device)
        self.to(self.device)

    def forward(self, x, sublayer_output):
        # Aggiungi l'output del sub-layer all'input (connessione residuale)
        added = x.to(self.device) + sublayer_output.to(self.device)
        # Normalizza il risultato (normalizzazione di livello)
        output = self.norm(added).to(self.device)
        return output
