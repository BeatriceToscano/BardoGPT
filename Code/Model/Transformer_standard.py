import torch
from torch import nn

from Model.Decoder import Decoder
from Model.Embedding_transformer_standard import MusicEmbedding
from Model.Encoder import Encoder


class Transformer(nn.Module):
    def __init__(self,
                 encoder_embed_size=512, encoder_num_params=1, encoder_heads=64, encoder_ff_dim= 512, encoder_dropout_rate=0.1, encoder_eps=1e-6, encoder_num_layers=3,
                 decoder_embed_size=512, decoder_num_params=1, decoder_heads=64, decoder_ff_dim= 512, decoder_dropout_rate=0.1, decoder_eps=1e-6, decoder_num_layers=3,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 vocab_size=388, max_seq_length=2048
                 ):
        super(Transformer, self).__init__()
        self.device = device
        self.encoder_embedding = MusicEmbedding(vocab_size, encoder_embed_size, max_seq_length).to(self.device)
        self.decoder_embedding = MusicEmbedding(vocab_size, decoder_embed_size, max_seq_length).to(self.device)
        self.encoders = nn.ModuleList([Encoder(encoder_embed_size, encoder_num_params, encoder_heads, encoder_ff_dim, encoder_dropout_rate, encoder_eps, device=self.device).to(self.device) for _ in range(encoder_num_layers)])
        self.decoders = nn.ModuleList([Decoder(decoder_embed_size, decoder_num_params, decoder_heads, decoder_ff_dim, decoder_dropout_rate, decoder_eps, device=self.device).to(self.device) for _ in range(decoder_num_layers)])
        self.linear = nn.Linear(decoder_embed_size * decoder_num_params, vocab_size).to(self.device)
        self.to(self.device)

    def forward(self, src, trg):
        src = self.encoder_embedding(src.reshape(1, -1))
        trg = self.decoder_embedding(trg.reshape(1, -1))

        # Passaggio attraverso gli encoder layers
        for encoder in self.encoders:
            src = encoder(src)

        # Passaggio attraverso i decoder layers
        for decoder in self.decoders:
            trg = decoder(trg, src)

        # Applicazione del layer lineare finale
        output = self.linear(trg)

        return output

    def save(self, path):
        torch.save(self.state_dict(), path)
