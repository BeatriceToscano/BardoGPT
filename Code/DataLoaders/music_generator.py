from pretty_midi import *


def generate_music(track, track_tempo=120):
    midi = pretty_midi.PrettyMIDI()
    instruments = {}
    pitch_indices = track[[col for col in track.columns if 'pitch' in col]]
    tempo_indices = track[[col for col in track.columns if 'tempo' in col]]
    velocity_indices = track[[col for col in track.columns if 'velocity' in col]]
    program_indices = track[[col for col in track.columns if 'program' in col]]
    for p, t, v, p in zip(program_indices.values, tempo_indices.values, velocity_indices.values, pitch_indices.values):
        in_id = np.argmax(p)
        if in_id not in instruments:
            instruments[in_id] = pretty_midi.Instrument(program=in_id)
        velocity = np.argmax(v)
        tempo = np.argmax(t)
        for i, note in enumerate(p):
            if note == 1:
                start = instruments[in_id].notes[-1].end if len(instruments[in_id].notes) > 0 else 0
                end = start + (60 / track_tempo) * (4 / (2 ** tempo))
                instruments[in_id].notes.append(pretty_midi.Note(velocity=velocity, pitch=i, start=start, end=end))  # (end - start) / (60 / tempo)=duration in beats=>end=start+duration*60/tempo
    for instrument in instruments.values():
        midi.instruments.append(instrument)
    midi.write('output.mid')
