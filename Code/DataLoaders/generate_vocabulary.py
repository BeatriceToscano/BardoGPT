import os
import json
from music21 import converter, note, chord
from tqdm import tqdm

from Variables import *


def main(use_markers=False):
    files = []
    for d, _, fs in os.walk(DATASET_PATH_FOLDER):
        for n in fs:
            if n.lower().endswith('.mid') or n.lower().endswith('.midi'):
                files.append(os.path.join(d, n))

    # Carica i dizionari esistenti se presenti, altrimenti inizializza a vuoto
    try:
        with open('event_to_idx.json', 'r') as f:
            event_to_idx = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        event_to_idx = {}

    try:
        with open('idx_to_event.json', 'r') as f:
            idx_to_event = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        idx_to_event = {}

    events = set(event_to_idx.keys())  # Inizia con gli eventi esistenti

    for midi_path in tqdm(files, desc="Processing MIDI files"):
        score = converter.parse(midi_path)

        for part in score.recurse():
            if isinstance(part, note.Note):
                # Converti la tupla in una stringa per la serializzazione JSON
                events.add(f"Note_{part.pitch.midi}_{part.duration.quarterLength}")
            elif isinstance(part, chord.Chord):
                # Usa ':' come separatore per i pitch degli accordi per evitare conflitti
                events.add(f"Chord_{'.'.join(str(n.pitch.midi) for n in part.notes)}_{part.duration.quarterLength}")
            elif isinstance(part, note.Rest):
                events.add(f"Rest_0_{part.duration.quarterLength}")
    if use_markers:
        events.add('START')
        events.add('END')
    # Aggiorna i dizionari solo con i nuovi eventi
    new_events = sorted(list(events - set(event_to_idx.keys())))
    starting_index = len(event_to_idx)
    event_to_idx.update({event: i for i, event in enumerate(new_events, start=starting_index)})
    idx_to_event.update({i: event for i, event in enumerate(new_events, start=starting_index)})

    # Salva i dizionari aggiornati
    with open(EVENT_TO_IDX_JSON_PATH, 'w') as f:
        json.dump(event_to_idx, f, indent=4)

    with open(IDX_TO_EVENT_JSON_PATH, 'w') as f:
        json.dump(idx_to_event, f, indent=4)

    print(f"Total unique events: {len(event_to_idx)}")


if __name__ == "__main__":
    main()
