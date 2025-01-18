import pandas as pd
from typing import Dict, Optional, Union

from data_preprocessing import DataPreprocessor
from eventframing.cols_schema import EventFrameColsSchema
from eventframing.eventframe import EventFrame
from eventframing.event_type import EventType


class AddStartEndEventsPreprocessor(DataPreprocessor):
    def __init__(self):
        pass

    def apply(self,data: Union[pd.DataFrame, 'EventFrame'],
              cols_schema: Optional[EventFrameColsSchema] = None) -> 'EventFrame':
        super()._check_apply_params(data, cols_schema)
        data, cols_schema = super()._get_data_and_cols_schema(data, cols_schema)

        dt_col = cols_schema.event_timestamp
        event_col = cols_schema.event_name
        user_id_col = cols_schema.user_id
        event_type_col = cols_schema.event_type
        event_type_index_col = cols_schema.event_type_index
        event_id_col = cols_schema.event_id

        data = data.sort_values(by=[user_id_col, dt_col])

        path_starts = data.groupby([user_id_col]).head(1).copy()
        path_ends = data.groupby([user_id_col]).tail(1).copy()

        path_starts[event_type_col] = EventType.PATH_START.value.name
        path_starts[event_type_index_col] = EventType.PATH_START.value.order
        path_starts[event_col] = EventType.PATH_START.value.name
        path_starts[event_id_col] = path_starts[user_id_col].astype(str) + '_' + EventType.PATH_START.value.name

        path_ends[event_type_col] = EventType.PATH_END.value.name
        path_ends[event_type_index_col] = EventType.PATH_END.value.order
        path_ends[event_col] = EventType.PATH_END.value.name
        path_ends[event_id_col] = path_ends[user_id_col].astype(str) + '_' + EventType.PATH_END.value.name

        # Объединяем оригинальные данные и новые события
        new_data = pd.concat([data, path_starts, path_ends], ignore_index=True)
        new_data = new_data.sort_values(by=[user_id_col, dt_col, event_type_index_col])

        # Возвращаем новый экземпляр EventFrame с обновленными данными
        return EventFrame(new_data, cols_schema)
