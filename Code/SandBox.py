import os

from mido import MidiFile

mid = MidiFile(r"H:\User\Desktop\BardoGPT\DataSets\Classical\alb_esp3.mid", clip=True)

message_numbers = []
duplicates = []

for track in mid.tracks:
    if len(track) in message_numbers:
        duplicates.append(track)
    else:
        message_numbers.append(len(track))

for track in duplicates:
    mid.tracks.remove(track)

mid.save(r"H:\User\Desktop\new_song.mid")