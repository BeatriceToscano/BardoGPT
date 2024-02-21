import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from DataLoaders.pretty_midi_DataLoader import DataLoader
from Model.Transformer import Transformer


def get_lr(step, model_size=512 * 6, factor=1, warmup=4000):
    # Calcola il learning rate con warmup e decadimento
    arg1 = step ** -0.5
    arg2 = step * (warmup ** -1.5)
    return factor * (model_size ** -0.5) * min(arg1, arg2)


def train(dl, model, warmup=4000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, betas=(0.9, 0.98), eps=1e-9)

    criterion1 = torch.nn.SmoothL1Loss()  # start, duration
    criterion2 = torch.nn.BCEWithLogitsLoss()  # isdrum
    criterion3 = torch.nn.BCEWithLogitsLoss()  # program
    criterion4 = torch.nn.BCEWithLogitsLoss()  # pitch
    criterion5 = torch.nn.BCEWithLogitsLoss()  # velocity
    losses = []
    losses1 = []
    losses2 = []
    losses3 = []
    losses4 = []
    losses5 = []
    matplotlib.use('TkAgg')
    fig, axs = plt.subplots(3, 2, figsize=(18, 12))
    fig.tight_layout()
    plt.ion()
    axs[0, 0].set_title('Loss')
    axs[0, 1].set_title('Loss1')
    axs[1, 0].set_title('Loss2')
    axs[1, 1].set_title('Loss3')
    axs[2, 0].set_title('Loss4')
    axs[2, 1].set_title('Loss5')
    matplotlib.use('TkAgg')
    step = 1
    for song in tqdm(dl):
        for track in song:
            lr = get_lr(step, warmup=warmup)
            step += 1

            # Aggiorna il learning rate per tutti i gruppi di parametri
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            reference = torch.Tensor(track.drop(track.head(1).index).values).to(model.device)

            optimizer.zero_grad()
            prediction = model(track)
            loss1 = criterion1(prediction[:, :2], reference[:, :2])
            loss2 = criterion2(prediction[:, 2:4], reference[:, 2:4])
            loss3 = criterion3(prediction[:, 4:132], reference[:, 4:132])
            loss4 = criterion4(prediction[:, 132:260], reference[:, 132:260])
            loss5 = criterion5(prediction[:, 260:], reference[:, 260:])
            loss = loss1 + loss2 + loss3 + loss4 + loss5
            loss.backward()
            optimizer.step()
            print(f"\rLoss: {loss}")
            losses.append(loss.detach().cpu().numpy())
            losses1.append(loss1.detach().cpu().numpy())
            losses2.append(loss2.detach().cpu().numpy())
            losses3.append(loss3.detach().cpu().numpy())
            losses4.append(loss4.detach().cpu().numpy())
            losses5.append(loss5.detach().cpu().numpy())
            for ax in axs.flat:
                ax.cla()
            axs[0, 0].plot(losses)
            axs[0, 1].plot(losses1)
            axs[1, 0].plot(losses2)
            axs[1, 1].plot(losses3)
            axs[2, 0].plot(losses4)
            axs[2, 1].plot(losses5)
            fig.show()
            plt.pause(0.00001)
        plt.savefig("losses.png")
    plt.close('all')
    return model


def test(model, max_length=1000):
    track = None
    model.eval()
    columns = ['start', 'duration', 'is_drum', 'is_not_drum']
    columns.extend([f'program{k}' for k in range(128)])
    columns.extend([f'pitch{k}' for k in range(128)])
    columns.extend([f'velocity{k}' for k in range(128)])
    sos = {
        'start': 0.0,
        'duration': 0.0,
        'is_drum': 0.0,
        'is_not_drum': 0.0
    }
    for k in range(128):
        sos[f'program{k}'] = 0.0
        sos[f'pitch{k}'] = 0.0
        sos[f'velocity{k}'] = 0.0
    start = pd.DataFrame([sos, sos], columns=columns)
    prediction = model(start)
    predicted_notes = [sos]
    for _ in tqdm(range(max_length)):
        predicted_note = {
            'start': prediction[-1, 0].item(),
            'duration': prediction[-1, 1].item(),
            'is_drum': float(prediction[-1, 2].item() > 0.5),
            'is_not_drum': float(prediction[-1, 3].item() > 0.5)
        }
        for k in range(128):
            predicted_note[f'program{k}'] = 1.0 if prediction[-1, 4 + k] > 0.6 else 0.0
            predicted_note[f'pitch{k}'] = 1.0 if prediction[-1, 132 + k] > 0.6 else 0.0
            predicted_note[f'velocity{k}'] = 1.0 if prediction[-1, 260 + k] > 0.6 else 0.0
        predicted_notes.append(predicted_note)
        track = pd.DataFrame(predicted_notes, columns=columns)
        if predicted_note["is_drum"] == predicted_note["is_not_drum"]:
            break
        prediction = model(track)
    return track


if __name__ == "__main__":
    dataLoader = DataLoader()
    transformer = Transformer()
    if os.path.exists("model.pth"):
        transformer.load_state_dict(torch.load("model.pth"))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    transformer = train(dataLoader, transformer, warmup=100)
    transformer.save("model.pth")
    print(test(transformer, 10))
