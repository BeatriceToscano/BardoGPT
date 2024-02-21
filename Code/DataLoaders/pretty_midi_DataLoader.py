import os
from collections import defaultdict

import mido
import numpy as np
import pandas as pd
import pretty_midi
from mido import MidiFile, parse_string

from Variables import DATASET_PATH_FOLDER


class DataLoader:
    def __init__(self, prepare=True):
        self.path = DATASET_PATH_FOLDER
        self.pointer = 0
        self.files = []
        self.prepare = prepare
        for d, _, fs in os.walk(self.path):
            for n in fs:
                if n.lower().endswith('.mid'):
                    self.files.append(os.path.join(d, n))

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
        mid = pretty_midi.PrettyMIDI(file)
        if prep:
            return self.prepareData(mid)
        return mid

    def prepareData(self, mid):
        instruments = []
        for instrument in mid.instruments:
            is_drum = int(instrument.is_drum)
            program = instrument.program
            notes = {}
            last_start, last_end, n = -1, -1, 0
            columns = ['start', 'duration', 'is_drum', 'is_not_drum']
            columns.extend([f'program{k}' for k in range(128)])
            columns.extend([f'pitch{k}' for k in range(128)])
            columns.extend([f'velocity{k}' for k in range(128)])
            # SOS
            notes[0] = {
                'start': 0.0,
                'duration': 0.0,
                'is_drum': 0.0,
                'is_not_drum': 0.0
            }
            for k in range(128):
                notes[0][f'program{k}'] = 0.0
                notes[0][f'pitch{k}'] = 0.0
                notes[0][f'velocity{k}'] = 0.0

            for note in instrument.notes:
                if last_start != note.start and last_end != note.end:
                    n += 1
                    notes[n] = {
                        'start': note.start,
                        'duration': note.end- note.start,
                        'is_drum': is_drum,
                        'is_not_drum': 1 - is_drum

                    }
                    for k in range(128):
                        notes[n][f'program{k}'] = 1.0 if k == program else 0.0
                        notes[n][f'pitch{k}'] = 0.0
                        notes[n][f'velocity{k}'] = 0.0

                last_start, last_end = note.start, note.end

                notes[n][f'pitch{note.pitch}'] = 1.0
                notes[n][f'velocity{note.velocity}'] = 1.0
            n += 1
            # EOS
            notes[n] = {
                'start': 0.0,
                'duration': 0.0,
                'is_drum': 1.0,
                'is_not_drum': 1.0
            }
            for k in range(128):
                notes[n][f'program{k}'] = 1.0
                notes[n][f'pitch{k}'] = 1.0
                notes[n][f'velocity{k}'] = 1.0
            frame = pd.DataFrame(notes).T
            frame = frame.reindex(columns=columns)
            instruments.append(frame)

        return instruments


"""
dl = DataLoader()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
for d in dl[0]:
    print(d)
    print()
"""
