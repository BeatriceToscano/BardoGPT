from DataLoaders.DataLoader import DataLoader

from rich.progress import Progress

from Variables import MESSAGES_DUMP_FILE


def main():
    dl = DataLoader()
    messages = []

    with Progress(auto_refresh=True) as progress:
        task1 = progress.add_task("[red]Song", total=len(dl))
        for i, mid in enumerate(dl):
            try:
                messages.extend(str(mid).split("\n"))

                progress.update(task1, advance=1, description=f"{i}/{len(dl)} " + mid.filename.ljust(200))
            except Exception as e:
                print(f"Error: {e}")
                progress.update(task1, advance=1, description=(f"{i}/{len(dl)} " + mid.filename).ljust(200))
                continue

    messages = list(set(messages))
    messages.sort()

    with open(MESSAGES_DUMP_FILE, "w") as f:
        for msg in messages:
            f.write(msg + "\n")


if __name__ == "__main__":
    main()
