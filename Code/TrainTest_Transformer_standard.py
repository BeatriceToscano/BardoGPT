import json
import os
from fractions import Fraction

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from music21 import note, chord, stream, midi
from tqdm import tqdm

from DataLoaders import generate_vocabulary
from DataLoaders.DataLoader_transformer_standard import DataLoader
from Model.Transformer_standard import Transformer
from Variables import IDX_TO_EVENT_JSON_PATH, EVENT_TO_IDX_JSON_PATH, MODEL_PATH, VALIDATION_PATH


def get_lr(step, model_size=512 * 6, factor=1, warmup=4000):
    # Calcola il learning rate con warmup e decadimento
    arg1 = step ** -0.5
    arg2 = step * (warmup ** -1.5)
    return factor * (model_size ** -0.5) * min(arg1, arg2)


def train(dl, model, warmup=4000, idx_to_event=None, event_to_idx=None, epochs=10, user_markers=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, betas=(0.9, 0.98), eps=1e-9)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    step = 0

    model.train()  # Imposta il modello in modalità training
    matplotlib.use('TkAgg')
    plt.figure()
    plt.ion()
    for epoch in range(epochs):
        for event_indices, _ in tqdm(dl):  # Ignora il secondo elemento (valori testuali)
            step += 1
            lr = get_lr(step, warmup=warmup)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()

            # Prepara gli input e i target per il modello
            event_indices = torch.tensor(event_indices, dtype=torch.long).to(model.device)
            src = event_indices[:-1]  # Input per il modello, escludi l'ultimo indice per src
            trg = event_indices[1:]  # Target per il modello, escludi il primo indice per trg

            prediction = model(src, trg)  # src utilizzato sia come input che come target iniziale per la predizione

            # Applica la CrossEntropyLoss
            prediction = prediction.view(-1, prediction.size(-1))
            trg = trg.contiguous().view(-1)
            loss = criterion(prediction, trg)
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())
            plt.clf()
            plt.plot(losses)
            plt.pause(0.001)
            plt.show()
            del loss, prediction, trg, event_indices, src
            torch.cuda.empty_cache()
            if step % 100 == 0:
                model.save("model.pth")
                test(model, idx_to_event, event_to_idx, save_path=os.path.join(VALIDATION_PATH, f"generated_{epoch}_{step}.mid"), use_markers=user_markers)
    plt.ioff()
    plt.savefig("losses.png")

    return model


def test(model, idx_to_event, event_to_idx, max_length=1000, start_sequence=None, temperature=1.0, save_path=None, use_markers=False):
    model.eval()  # Imposta il modello in modalità valutazione

    if start_sequence is None:
        start_sequence = [torch.randint(0, len(event_to_idx), (1,)).item()]  # Genera un evento casuale come seed
    if use_markers:
        start_sequence = [event_to_idx['START']]
    sequence = start_sequence

    input_sequence = start_sequence
    output_sequence = []
    for _ in range(max_length):
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(model.device)
        with torch.no_grad():
            output_logits = model(input_tensor, input_tensor)
            output_probabilities = torch.softmax(output_logits[:, -1, :] / temperature, dim=-1)
            predicted_index = torch.multinomial(output_probabilities, 1).item()

        input_sequence = [predicted_index]
        output_sequence.append(predicted_index)
        if use_markers and predicted_index == event_to_idx['END']:
            break
    # Converti gli indici in eventi musicali
    events = [idx_to_event[str(idx)] for idx in output_sequence]

    # Crea una partitura musicale con music21
    s = stream.Stream()

    for event in events:
        if event.startswith('Note'):
            pitch, duration = event.split('_')[1:]
            s.append(note.Note(int(pitch), quarterLength=float(Fraction(duration))))
        elif event.startswith('Chord'):
            pitches, duration = event.split('_')[1:]
            s.append(chord.Chord([int(pitch) for pitch in pitches.split('.')], quarterLength=float(Fraction(duration))))
        elif event.startswith('Rest'):
            _, _, duration = event.split('_')
            s.append(note.Rest(quarterLength=float(Fraction(duration))))

    # Salva la partitura come file MIDI
    if save_path is not None:
        midi_path = save_path
    else:
        midi_path = "generated_music.mid"
    mf = midi.translate.streamToMidiFile(s)
    mf.open(midi_path, 'wb')
    mf.write()
    mf.close()

    print(f"Generated music saved to {midi_path}")
    return events


def main():
    generate_vocabulary.main(use_markers=True)
    with open(EVENT_TO_IDX_JSON_PATH, 'r') as f:
        event_to_idx = json.load(f)

    with open(IDX_TO_EVENT_JSON_PATH, 'r') as f:
        idx_to_event = json.load(f)
    dataLoader = DataLoader(use_markers=True)
    transformer = Transformer(vocab_size=len(event_to_idx), max_seq_length=2048)
    if os.path.exists(MODEL_PATH):
        transformer.load_state_dict(torch.load(MODEL_PATH))

    transformer = train(dataLoader, transformer, warmup=1000, idx_to_event=idx_to_event, epochs=2, user_markers=True)
    transformer.save("model.pth")
    print(np.unique(test(transformer, idx_to_event, event_to_idx, 100, use_markers=True), return_counts=True))


if __name__ == "__main__":
    main()
