import os

basepath = os.getcwd()

DATASET_PATH_FOLDER = os.path.join(basepath, "DataSets", "Classical")
MESSAGES_DUMP_FILE = os.path.join(basepath, "Dumps", "messages_dump.txt")
EVENT_TO_IDX_JSON_PATH = os.path.join(basepath, "event_to_idx.json")
IDX_TO_EVENT_JSON_PATH = os.path.join(basepath, "idx_to_event.json")
MODEL_PATH = os.path.join(basepath, "model.pth")
VALIDATION_PATH = os.path.join(basepath, "Validation")
