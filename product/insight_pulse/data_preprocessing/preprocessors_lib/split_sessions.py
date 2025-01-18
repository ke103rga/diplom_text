import pandas as pd
from typing import Dict, List, Union, Optional, Tuple

from data_preprocessing import DataPreprocessor
from eventframing.eventframe import EventFrame
from eventframing.cols_schema import EventFrameColsSchema
from eventframing.event_type import EventType
from utils.TimeUnits import TimeUnits


class SplitSessionsPreprocessor(DataPreprocessor):
    session_id_col = 'session_id'

    def __init__(self, timeout: Optional[Union[int, 'TimeUnits', Tuple[int, str]]], split_events: Optional[List[str]] = None):
        if sum([1 if param is not None else 0 for param in [timeout, split_events]]) != 1:
            raise ValueError('exactly 1 param')

        if isinstance(timeout, int):
            self.timeout = pd.Timedelta(minutes=timeout)
        elif isinstance(timeout, tuple):
            self.timeout = TimeUnits(timeout).get_time_delta()
        elif isinstance(timeout, TimeUnits):
            self.timeout = timeout.get_time_delta()
        else:
            raise ValueError('int or TimeUnits')
        self.split_events = split_events

    def apply(self, data: Union[pd.DataFrame, 'EventFrame'],
              cols_schema: Optional[EventFrameColsSchema] = None) -> 'EventFrame':

        super()._check_apply_params(data, cols_schema)
        data, cols_schema = super()._get_data_and_cols_schema(data, cols_schema)

        if self.timeout is not None:
            split_data = self._split_by_timeout(data, cols_schema)

        split_data = self._add_synthetic_events(split_data, cols_schema)
        split_data = split_data.sort_values(by=[
            cols_schema.user_id,
            cols_schema.event_timestamp,
            cols_schema.event_type_index
        ])
        cols_schema.session_id = self.session_id_col

        return EventFrame(split_data, cols_schema)

    def _split_by_timeout(self, data: pd.DataFrame, cols_schema: EventFrameColsSchema) -> pd.DataFrame:
        dt_col = cols_schema.event_timestamp
        user_id_col = cols_schema.user_id

        data['time_from_last_event'] = data[dt_col] - data.groupby(user_id_col)[dt_col].shift()
        data['new_session_start'] = data['time_from_last_event'] > self.timeout
        data[self.session_id_col] = data[user_id_col].astype(str) + '_' + data['new_session_start'].cumsum().astype(str)

        return data.drop(columns=['time_from_last_event', 'new_session_start'])

    def _add_synthetic_events(self, data: pd.DataFrame, cols_schema: EventFrameColsSchema) -> pd.DataFrame:
        event_col = cols_schema.event_name
        event_type_col = cols_schema.event_type
        event_type_index_col = cols_schema.event_type_index
        event_id_col = cols_schema.event_id

        session_starts = data.groupby([self.session_id_col]).head(1).copy()
        session_ends = data.groupby([self.session_id_col]).tail(1).copy()

        session_starts[event_type_col] = EventType.SESSION_START.value.name
        session_starts[event_type_index_col] = EventType.SESSION_START.value.order
        session_starts[event_col] = EventType.SESSION_START.value.name
        session_starts[event_id_col] = session_starts[self.session_id_col].astype(str) + EventType.SESSION_START.value.name

        session_ends[event_type_col] = EventType.SESSION_END.value.name
        session_ends[event_type_index_col] = EventType.SESSION_END.value.order
        session_ends[event_col] = EventType.SESSION_END.value.name
        session_ends[event_id_col] = session_ends[self.session_id_col].astype(str) + EventType.SESSION_END.value.name

        preprocessed_data = pd.concat([data, session_starts, session_ends])
        return preprocessed_data

