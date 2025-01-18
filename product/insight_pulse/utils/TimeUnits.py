import pandas as pd
from typing import Union, Tuple


class TimeUnits:
    TIME_UNITS = ["Y", "M", "W", "D", "h", "m", "s", "ms",]
    TIME_UNITS_SET = set(TIME_UNITS)

    def __init__(self, time_tuple: Union[Tuple[int, str], 'TimeUnits']):
        if isinstance(time_tuple, tuple):
            if isinstance(time_tuple[0], int) and time_tuple[0] > 0:
                self.quantity = time_tuple[0]
            else:
                raise ValueError(f'Quantity is integer positive!')
            if time_tuple[1] in self.TIME_UNITS_SET:
                self.time_unit = time_tuple[1]
            else:
                raise ValueError(f'One of {self.TIME_UNITS}')
        elif isinstance(time_tuple, TimeUnits):
            self.quantity = time_tuple.quantity
            self.time_unit = time_tuple.time_unit

    def get_time_delta(self):
        """
        Преобразует кортеж (quantity, time_unit) в pd.Timedelta.
        :return: pd.Timedelta, представляющий указанное время.
        """
        # Определяем единицу времени
        if self.time_unit == 's':  # секунды
            return pd.Timedelta(seconds=self.quantity)
        elif self.time_unit == 'm':  # минуты
            return pd.Timedelta(minutes=self.quantity)
        elif self.time_unit == 'h':  # часы
            return pd.Timedelta(hours=self.quantity)
        elif self.time_unit == 'D':  # дни
            return pd.Timedelta(days=self.quantity)
        elif self.time_unit == 'W':  # недели
            return pd.Timedelta(weeks=self.quantity)
        elif self.time_unit == 'M':  # месяцы
            return pd.Timedelta(months=self.quantity)
        elif self.time_unit == 'Y':  # года
            return pd.Timedelta(years=self.quantity)