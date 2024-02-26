import os
import json
from music21 import converter, note, chord
from tqdm import tqdm

from Variables import *


class DataLoader:
    def __init__(self, prepare=True, use_markers=False):
        self.use_markers = use_markers
        self.path = DATASET_PATH_FOLDER
        self.pointer = 0
        self.files = []
        self.prepare = prepare
        # Carica il vocabolario
        self.event_to_idx, self.idx_to_event = self.load_vocabulary(EVENT_TO_IDX_JSON_PATH, IDX_TO_EVENT_JSON_PATH)

        for d, _, fs in os.walk(self.path):
            for n in fs:
                if n.lower().endswith('.mid') or n.lower().endswith('.midi'):
                    self.files.append(os.path.join(d, n))

    def load_vocabulary(self, event_to_idx_path, idx_to_event_path):
        with open(event_to_idx_path, 'r') as f:
            event_to_idx = json.load(f)
        with open(idx_to_event_path, 'r') as f:
            idx_to_event = {int(k): v for k, v in json.load(f).items()}
        return event_to_idx, idx_to_event

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.getDataAbout(self.files[idx], self.prepare)

    def __iter__(self):
        self.pointer = 0
        return self

    def __next__(self):
        if self.pointer < len(self.files):
            self.pointer += 1
            return self.getDataAbout(self.files[self.pointer - 1], self.prepare)
        else:
            self.pointer = 0
            raise StopIteration

    def getDataAbout(self, file, prep=False):
        score = converter.parse(file)
        if prep:
            return self.prepareData(score)
        return score

    def prepareData(self, score):
        event_indices = []
        event_names = []
        keys = 0
        if self.use_markers:
            event_indices.append(self.event_to_idx['START'])
            event_names.append('START')
        for part in score.recurse():
            keys += 1
            if keys+2 >= 2048:
                break
            if isinstance(part, note.Note):
                event_key = f"Note_{part.pitch.midi}_{part.duration.quarterLength}"
            elif isinstance(part, chord.Chord):
                pitches = ".".join(str(n.midi) for n in part.pitches)
                event_key = f"Chord_{pitches}_{part.duration.quarterLength}"
            elif isinstance(part, note.Rest):
                event_key = f"Rest_0_{part.duration.quarterLength}"
            else:
                continue  # Ignora altri tipi di eventi

            if event_key in self.event_to_idx:
                event_indices.append(self.event_to_idx[event_key])
                event_names.append(event_key)
            else:
                # Gestisci il caso in cui l'evento non sia nel vocabolario
                print(f"Evento non trovato nel vocabolario: {event_key}")
        if self.use_markers:
            event_indices.append(self.event_to_idx['END'])
            event_names.append('END')
        return event_indices, event_names
