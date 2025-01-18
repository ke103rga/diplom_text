from enum import Enum


class EventTypeDesc:
    name: str
    order: int

    def __init__(self, name: str, order: int):
        self.name = name
        self.order = order


class EventType(Enum):
    RAW = EventTypeDesc("raw", 2)
    PATH_START = EventTypeDesc("path_start", 0)
    PATH_END = EventTypeDesc("path_end", 5)
    SESSION_START = EventTypeDesc("session_start", 1)
    SESSION_END = EventTypeDesc("session_end", 3)
