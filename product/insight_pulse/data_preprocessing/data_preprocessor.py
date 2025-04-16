from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd

from eventframing.eventframe import EventFrame
from eventframing.cols_schema import EventFrameColsSchema


class DataPreprocessor(ABC):
    """
    An interface for data preprocessors.
    """

    @abstractmethod
    def __init__(self, **kwargs: dict) -> None:
        """
        Initializes the data preprocessor with variable arguments.

        Args:
            kwargs: Variable keyword arguments for specific preprocessor settings.
        """
        pass

    @abstractmethod
    def apply(self, data: 'EventFrame', prepare: bool) -> 'EventFrame':
        """
        Applies a transformation to an EventFrame object.

        Args:
            data (EventFrame): The EventFrame data to transform.
            prepare (bool): Indicates whether the transformation should prepare the data.

        Returns:
            EventFrame: The transformed EventFrame object.
        """
        pass

    def _check_apply_params(self, data: 'EventFrame') -> None:
        """
        Checks the parameters for the apply method.

        Args:
            data (EventFrame): The EventFrame object to check.

        Raises:
            ValueError: If the data is not an instance of EventFrame.
        """
        if not isinstance(data, EventFrame):
            raise ValueError('data should be EventFrame')

    def _get_data_and_cols_schema(self, data: 'EventFrame', prepare: bool) ->\
            Tuple[pd.DataFrame, 'EventFrameColsSchema']:
        """
        Retrieves the data and column schema from an EventFrame.

        Args:
            data (EventFrame): The EventFrame object to extract data from.
            prepare (bool): Indicates whether to prepare the data.

        Returns:
            Tuple[pd.DataFrame, EventFrameColsSchema]: A tuple containing a copy of the data and its column schema.

        Raises:
            ValueError: If the data is not an instance of EventFrame.
        """
        if isinstance(data, EventFrame):
            return data.data.copy(), data.cols_schema
        else:
            raise ValueError('data should be EventFrame')
