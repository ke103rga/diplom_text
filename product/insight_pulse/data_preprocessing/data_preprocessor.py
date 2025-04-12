from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, List, Union, Tuple

from eventframing.eventframe import EventFrame
from eventframing.cols_schema import EventFrameColsSchema


class DataPreprocessor(ABC):
    """
    Интерфейс для данных препроцессоров.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Инициализация с использованием аргументов с переменным числом.
        """
        pass

    @abstractmethod
    def apply(self, data: EventFrame) -> 'EventFrame':
        """
        Применяет некоторое преобразование к объекту EventFrame.
        """
        pass

    #TODO: delete cols_schema param from methods below

    def _check_apply_params(self, data: EventFrame,
                            cols_schema: Union[Dict[str, str], EventFrameColsSchema]) -> None:
        if not isinstance(data, EventFrame):
            raise ValueError('data should be EventFrame')
            # if isinstance(data, pd.DataFrame):
            #     if cols_schema is None:
            #         raise ValueError('You should pass cols schema with data in DataFrame')
            #     elif not isinstance(cols_schema, EventFrameColsSchema) and not isinstance(cols_schema, dict):
            #         raise ValueError('cols_schema should be EventFrameColsSchema or dict')
            # else:
            #     raise ValueError('EventFrame or DataFrame with EventFrameColsSchema')

    def _get_data_and_cols_schema(self, data: EventFrame,
                            cols_schema: Union[Dict[str, str], EventFrameColsSchema],
                            prepare: bool) -> Tuple[pd.DataFrame, EventFrameColsSchema]:
        if isinstance(data, EventFrame):
            return data.data.copy(), data.cols_schema
        else:
            ef = EventFrame(data, cols_schema, prepare=prepare)
            return ef.to_dataframe(), ef.cols_schema
