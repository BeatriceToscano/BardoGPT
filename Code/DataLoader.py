import os

from Variables import DATASET_PATH_FOLDER


class DataLoader:
    def __init__(self):
        self.path = DATASET_PATH_FOLDER
        self.pointer = 0
        self.files = []
        for d, _, fs in os.walk(self.path):
            for n in fs:
                if n.lower().endswith('.mid'):
                    self.files.append(os.path.join(d, n))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.getDataAbout(self.files[idx])

    def __iter__(self):
        self.pointer = 0
        return self

    def __next__(self):
        if self.pointer < len(self.files):
            self.pointer += 1
            return self[self.pointer - 1]
        else:
            self.pointer = 0
            raise StopIteration

    def getDataAbout(self, file):
        return file
