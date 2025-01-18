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
    def apply(self, data: Union[pd.DataFrame, 'EventFrame'],
              cols_schema: Optional[EventFrameColsSchema] = None) -> 'EventFrame':
        """
        Применяет некоторое преобразование к объекту EventFrame.
        """
        pass

    def _check_apply_params(self, data: Union[pd.DataFrame, 'EventFrame'],
                            cols_schema: Optional[EventFrameColsSchema]) -> None:
        if not (isinstance(data, EventFrame) or
                (isinstance(data, pd.DataFrame) and isinstance(cols_schema, EventFrameColsSchema))):
            raise ValueError('EventFrame or DataFrame with EventFrameColsSchema')

    def _get_data_and_cols_schema(self, data: Union[pd.DataFrame, 'EventFrame'],
                            cols_schema: Optional[EventFrameColsSchema]) -> Tuple[pd.DataFrame, EventFrameColsSchema]:
        if isinstance(data, EventFrame):
            return data.data, data.cols_schema
        else:
            return data, cols_schema
