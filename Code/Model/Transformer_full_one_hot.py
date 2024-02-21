import torch
from torch import nn

from Model.Decoder import Decoder
from Model.Embedding_full_one_hot import MusicEmbedding
from Model.Encoder import Encoder


class Transformer(nn.Module):
    def __init__(self,
                 encoder_embed_size=512, encoder_num_params=5, encoder_heads=64, encoder_ff_dim=12 * 512, encoder_dropout_rate=0.1, encoder_eps=1e-6, encoder_num_layers=6,
                 decoder_embed_size=512, decoder_num_params=5, decoder_heads=64, decoder_ff_dim=12 * 512, decoder_dropout_rate=0.1, decoder_eps=1e-6, decoder_num_layers=6,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 output_linear_size=128 * 3 + 2 + 8
                 ):
        super(Transformer, self).__init__()
        self.device = device
        self.encoder_embedding = MusicEmbedding(encoder_embed_size).to(self.device)
        self.decoder_embedding = MusicEmbedding(decoder_embed_size).to(self.device)
        self.encoders = nn.ModuleList([Encoder(encoder_embed_size, encoder_num_params, encoder_heads, encoder_ff_dim, encoder_dropout_rate, encoder_eps).to(self.device) for _ in range(encoder_num_layers)])
        self.decoders = nn.ModuleList([Decoder(decoder_embed_size, decoder_num_params, decoder_heads, decoder_ff_dim, decoder_dropout_rate, decoder_eps).to(self.device) for _ in range(decoder_num_layers)])
        self.linear = nn.Linear(decoder_embed_size * decoder_num_params, output_linear_size).to(self.device)
        self.to(self.device)

    def forward(self, track):
        embeddings = []
        for i, note in enumerate(track.drop(track.tail(1).index).iloc):
            pitch_indices = torch.tensor(track[[col for col in track.columns if 'pitch' in col]].iloc[i].values, dtype=torch.int).to(self.device)
            tempo_indices = torch.tensor(track[[col for col in track.columns if 'tempo' in col]].iloc[i].values, dtype=torch.int).to(self.device)
            velocity_indices = torch.tensor(track[[col for col in track.columns if 'velocity' in col]].iloc[i].values, dtype=torch.int).to(self.device)
            program_indices = torch.tensor(track[[col for col in track.columns if 'program' in col]].iloc[i].values, dtype=torch.int).to(self.device)
            drum_indices = torch.tensor(track[[col for col in track.columns if 'drum' in col]].iloc[i].values, dtype=torch.int).to(self.device)

            combined_embeddings = self.encoder_embedding(pitch_indices, velocity_indices, program_indices, tempo_indices, drum_indices).to(self.device)
            embeddings.append(combined_embeddings)
        combined_embeddings = torch.cat(embeddings, dim=0).reshape(len(track) - 1, embeddings[0].shape[0], embeddings[0].shape[1]).to(self.device)
        encoder_output = combined_embeddings.to(self.device)
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output).to(self.device)

        embeddings = []
        for i, note in enumerate(track.drop(track.head(1).index).iloc):
            pitch_indices = torch.tensor(track[[col for col in track.columns if 'pitch' in col]].iloc[i].values, dtype=torch.int).to(self.device)
            tempo_indices = torch.tensor(track[[col for col in track.columns if 'tempo' in col]].iloc[i].values, dtype=torch.int).to(self.device)
            velocity_indices = torch.tensor(track[[col for col in track.columns if 'velocity' in col]].iloc[i].values, dtype=torch.int).to(self.device)
            program_indices = torch.tensor(track[[col for col in track.columns if 'program' in col]].iloc[i].values, dtype=torch.int).to(self.device)
            drum_indices = torch.tensor(track[[col for col in track.columns if 'drum' in col]].iloc[i].values, dtype=torch.int).to(self.device)

            combined_embeddings = self.decoder_embedding(pitch_indices, velocity_indices, program_indices, tempo_indices, drum_indices).to(self.device)
            embeddings.append(combined_embeddings)
        combined_embeddings = torch.cat(embeddings, dim=0).reshape(len(track) - 1, embeddings[0].shape[0], embeddings[0].shape[1]).to(self.device)

        decoder_output = combined_embeddings
        for decoder in self.decoders:
            decoder_output = decoder(decoder_output, encoder_output).to(self.device)
        output = []
        for i in range(len(decoder_output)):
            opt = self.linear(decoder_output[i]).to(self.device)
            output.append(opt)
            opt[:, 0:2] = torch.sigmoid(opt[:, 0:2])
            opt[:, 2:10] = torch.softmax(opt[:, 2:10], dim=1)
            opt[:, 10:138] = torch.softmax(opt[:, 10:138], dim=1)
            opt[:, 138:266] = torch.sigmoid(opt[:, 138:266])
            opt[:, 266:] = torch.softmax(opt[:, 266:], dim=1)
        return torch.cat(output).to(self.device)

    def save(self, path):
        torch.save(self.state_dict(), path)
