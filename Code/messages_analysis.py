import os

from DataLoader import DataLoader
from mido import MidiFile

from rich.progress import Progress

from Variables import MESSAGES_DUMP_FILE


def main():
    dl = DataLoader()
    messages = {}

    with open(MESSAGES_DUMP_FILE, "w") as f:
        with Progress() as progress:
            task1 = progress.add_task("[red]Song", total=len(dl))
            for song in dl:

                try:
                    mid = MidiFile(song)
                    task2 = progress.add_task("[green]Track".rjust(100, ' '), total=len(mid.tracks))
                    for track in mid.tracks:

                        task3 = progress.add_task("[cyan]Message".rjust(100, ' '), total=len(track))
                        for msg in track:
                            if msg.type in messages:
                                messages[msg.type] += 1
                            else:
                                messages[msg.type] = 1

                            progress.update(task3, advance=1, description=msg)

                        progress.remove_task(task3)
                        progress.update(task2, advance=1)
                    progress.remove_task(task2)
                    progress.update(task1, advance=1, description="[red]Song".rjust(100, ' ') + "\n" + "[white]" + song[-100:].ljust(100, ' '))
                except Exception as e:
                    progress.update(task1, advance=1, description="[red]Song".rjust(100, ' ') + "\n" + "[white]" + song[-100:].ljust(100, ' '))
                    try:
                        os.remove(song)
                    except:
                        f.write(f"{song} Error: {e}\n")
                        continue
                    continue
        progress.remove_task(task1)

        for k, v in messages.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
