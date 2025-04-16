import pandas as pd
from typing import Union

from data_preprocessing import DataPreprocessor
from eventframing.cols_schema import EventFrameColsSchema
from eventframing.eventframe import EventFrame
from eventframing.event_type import EventType


class AddStartEndEventsPreprocessor(DataPreprocessor):
    """
    A preprocessor that adds start and end events to the event stream.
    """

    def __init__(self) -> None:
        """
        Initializes the AddStartEndEventsPreprocessor.
        """
        pass

    def apply(self, data: Union[pd.DataFrame, 'EventFrame'], prepare: bool = True) -> 'EventFrame':
        """
        Applies the preprocessor to the given data, adding start and end events.

        Args:
            data (Union[pd.DataFrame, EventFrame]): The event data to process, either as a DataFrame or an EventFrame.
            prepare (bool, optional): A flag indicating whether to prepare the data. Defaults to True.

        Returns:
            EventFrame: The processed EventFrame object with start and end path events added.

        Raises:
            ValueError: If the input data is not an instance of EventFrame.
        """
        # Check and get data
        super()._check_apply_params(data)
        data, cols_schema = super()._get_data_and_cols_schema(data, prepare)

        # Specify columns in data
        dt_col = cols_schema.event_timestamp
        event_col = cols_schema.event_name
        user_id_col = cols_schema.user_id
        event_type_col = cols_schema.event_type
        event_type_index_col = cols_schema.event_type_index
        event_id_col = cols_schema.event_id

        # Sort values and delete already existing path_start and path_end events
        data = data.sort_values(by=[user_id_col, dt_col])
        data = self._delete_synthetic_events_if_exists(data, cols_schema)

        # Select first and last events as starts and ends of user's paths
        path_starts = data.groupby([user_id_col]).head(1).copy()
        path_ends = data.groupby([user_id_col]).tail(1).copy()

        # Change values in service cols
        path_starts[event_type_col] = EventType.PATH_START.value.name
        path_starts[event_type_index_col] = EventType.PATH_START.value.order
        path_starts[event_col] = EventType.PATH_START.value.name
        path_starts[event_id_col] = path_starts[user_id_col].astype(str) + '_' + EventType.PATH_START.value.name

        path_ends[event_type_col] = EventType.PATH_END.value.name
        path_ends[event_type_index_col] = EventType.PATH_END.value.order
        path_ends[event_col] = EventType.PATH_END.value.name
        path_ends[event_id_col] = path_ends[user_id_col].astype(str) + '_' + EventType.PATH_END.value.name

        # Concatenate new data with old events
        new_data = pd.concat([data, path_starts, path_ends], ignore_index=True)
        new_data = new_data.sort_values(by=[user_id_col, dt_col, event_type_index_col])

        return EventFrame(new_data, cols_schema)

    def _delete_synthetic_events_if_exists(self, data: pd.DataFrame, cols_schema: 'EventFrameColsSchema') ->\
            pd.DataFrame:
        """
        Deletes synthetic events from the data if they exist.

        Args:
            data (pd.DataFrame): The DataFrame containing events.
            cols_schema (EventFrameColsSchema): Schema for the EventFrame columns.

        Returns:
            pd.DataFrame: The DataFrame with synthetic events removed.
        """
        event_type_col = cols_schema.event_type
        event_types_to_delete = [EventType.PATH_START.value.name, EventType.PATH_END.value.name]
        return data[~data[event_type_col].isin(event_types_to_delete)]
