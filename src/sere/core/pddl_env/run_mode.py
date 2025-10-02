from enum import Enum

class RunMode(str, Enum):
    INTERACTIVE = "interactive"
    BATCH = "batch"
    OPEN_LOOP = "open_loop"
