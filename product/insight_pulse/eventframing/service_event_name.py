from enum import Enum


class ServiceEventName(Enum):
    PATH_START = "path_start"
    PATH_END = "path_end"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
