import pandas as pd
from typing import Dict, Optional, List, Union

from eventframing.cols_schema import EventFrameColsSchema
from eventframing.event_type import EventType


class EventFrame:

    def __init__(self, data: pd.DataFrame, cols_schema:  Union[Dict[str, str], 'EventFrameColsSchema'],
                 custom_cols: Optional[List] = None, prepare: bool = True):
        """
        Инициализация класса EventFrame.

        """
        self.data = data.copy()
        self.cols_schema = EventFrameColsSchema(cols_schema)
        if prepare:
            self.prepare()
        else:
            if cols_schema.event_id is None or cols_schema.event_type is None or cols_schema.event_type_index is None:
                raise ValueError('...')

    def prepare(self):
        if self.cols_schema.event_id is None:
            self._add_event_id()
        if self.cols_schema.event_type is None:
            self._add_event_type()
        if self.cols_schema.event_type_index is None:
            self._add_event_type_index()

        self.data[self.cols_schema.event_timestamp] = pd.to_datetime(self.data[self.cols_schema.event_timestamp])

    def to_dataframe(self) -> pd.DataFrame:
        """
        Получить данные DataFrame.
        """
        sort_cols_list = [self.cols_schema.user_id, self.cols_schema.event_timestamp]
        if self.cols_schema.event_type_index is not None:
            sort_cols_list.append(self.cols_schema.event_type_index)
            
        return self.data.copy().sort_values(sort_cols_list)

    def filter(self, conditions: List[str], inplace: bool = False) -> 'EventFrame':
        """
        Фильтрует строки DataFrame по заданным условиям.

        :param inplace:
        :param conditions: Список строковых выражений, представляющих условия фильтрации.
        :return: Новый объект EventFrame с отфильтрованными данными.
        """
        # Объединяем условия в одно для фильтрации
        combined_condition = " & ".join(conditions)

        # Фильтруем данные
        filtered_data = self.data.query(combined_condition)

        if inplace:
            self.data = filtered_data

        # Возвращаем новый объект EventFrame с отфильтрованными данными
        return EventFrame(filtered_data, self.cols_schema)

    def add_col(self, col_name: str, col_data: Union[pd.Series, List]):
        if len(col_data) != self.data.shape[0]:
            raise ValueError('...')

        self.data[col_name] = col_data

    def get_data_shape(self):
        return self.data.shape

    def copy(self) -> 'EventFrame':
        """
        Создает глубокую копию объекта EventFrame.

        :return: Новый экземпляр EventFrame, который является копией текущего.
        """
        copied_data = self.data.copy()  # Создаем глубокую копию DataFrame
        copied_cols_schema = self.cols_schema.copy()  # Если cols_schema не изменяется, просто копируем ссылку

        return EventFrame(copied_data, copied_cols_schema)

    def _add_event_type(self) -> None:
        data_len = self.data.shape[0]
        event_types = [EventType.RAW.value.name] * data_len
        event_type_col_name = 'event_type'

        self.add_col(event_type_col_name, event_types)
        self.cols_schema.event_type = event_type_col_name

    def _add_event_type_index(self) -> None:
        data_len = self.data.shape[0]
        event_type_idx = [EventType.RAW.value.order] * data_len
        event_type_index_col_name = 'event_type_index'

        self.add_col(event_type_index_col_name, event_type_idx)
        self.cols_schema.event_type_index = event_type_index_col_name

    def _add_event_id(self) -> None:
        # TODO: make id by hash of necessary columns for every new string
        data_len = self.data.shape[0]
        event_ids = list(range(data_len))
        event_id_col_name = 'event_id'

        self.add_col(event_id_col_name, event_ids)
        self.cols_schema.event_id = event_id_col_name

    def __repr__(self):
        """
        Строковое представление объекта EventFrame.
        """
        return f"EventFrame(data={self.data.shape[0]} rows, columns={self.data.columns.tolist()})"

