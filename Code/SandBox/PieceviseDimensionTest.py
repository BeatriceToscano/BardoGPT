import numpy as np
import torch

from DataLoaders.pretty_midi_DataLoader import DataLoader
from Model.AddNorm import AddNorm
from Model.Decoder import Decoder
from Model.Embedding import MusicEmbedding
from Model.Encoder import Encoder
from Model.MultiHeadAttention import MultiHeadAttention
from Model.PositionWiseFeedForward import PositionWiseFeedforward
from Model.Transformer import Transformer


def main():
    dl = DataLoader()
    track = dl[10001][3]
    embeddings = []
    for i, note in enumerate(track.iloc):
        pitch_indices = torch.tensor(track[[col for col in track.columns if 'pitch' in col]].iloc[i].values, dtype=torch.int)
        velocity_indices = torch.tensor(track[[col for col in track.columns if 'velocity' in col]].iloc[i].values, dtype=torch.int)
        program_indices = torch.tensor(track[[col for col in track.columns if 'program' in col]].iloc[i].values, dtype=torch.int)
        drum_indices = torch.tensor(track[[col for col in track.columns if 'drum' in col]].iloc[i].values, dtype=torch.int)
        continuous_features = torch.tensor(track[[col for col in track.columns if col in ["start", "duration"]]].iloc[i].values, dtype=torch.float)

        embedding_layer = MusicEmbedding()
        combined_embeddings = embedding_layer(pitch_indices, velocity_indices, program_indices, continuous_features, drum_indices)
        embeddings.append(combined_embeddings)
    combined_embeddings = torch.cat(embeddings, dim=0).reshape(len(track), embeddings[0].shape[0], embeddings[0].shape[1])

    print("Dimensione dell'embedding combinato:", combined_embeddings.shape)

    multiHead = MultiHeadAttention()
    mha = multiHead(combined_embeddings, combined_embeddings, combined_embeddings)
    print("Dimensione dell'attenzione:", mha.shape)
    add_and_norm = AddNorm()
    an = add_and_norm(combined_embeddings, mha)
    print("Dimensione dell'output di AddNorm:", an.shape)
    ff = PositionWiseFeedforward()
    output = ff(an)
    print("Dimensione dell'output di PositionWiseFeedforward:", output.shape)
    enc = Encoder()
    result = enc(combined_embeddings)
    print("Dimensione dell'output dell'encoder:", result.shape)
    dec = Decoder()
    result = dec(combined_embeddings, result)
    print("Dimensione dell'output del decoder:", result.shape)
    transformer = Transformer()
    result = transformer(track)
    print("Dimensione dell'output del transformer:", result.shape)
    for name, param in transformer.named_parameters():
        print(f"{name}: {param.size()}")


if __name__ == "__main__":
    main()
