import os

from music21 import converter, note, chord
from tqdm import tqdm

from Variables import DATASET_PATH_FOLDER
files=[]
for d, _, fs in os.walk(DATASET_PATH_FOLDER):
    for n in fs:
        if n.lower().endswith('.mid'):
            files.append(os.path.join(d, n))

    # Esempio per estrarre note e accordi
events = []
for midi_path in tqdm(files):
    score = converter.parse(midi_path)

    for part in score.recurse():
        if isinstance(part, note.Note):
            events.append(('Note', part.pitch.midi, part.duration.quarterLength))
        elif isinstance(part, chord.Chord):
            events.append(('Chord', '+'.join(str(n) for n in part.notes), part.duration.quarterLength))
        elif isinstance(part, note.Rest):
            events.append(('Rest', part.duration.quarterLength))
    # Aggiungi qui altri tipi di eventi se necessario
event_to_id = {event: i for i, event in enumerate(set(events))}
id_to_event = {i: event for event, i in event_to_id.items()}
tokenized_sequence = [event_to_id[event] for event in events]
print(event_to_id)