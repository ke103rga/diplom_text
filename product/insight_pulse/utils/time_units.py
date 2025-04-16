import pandas as pd
from typing import Union, Tuple , Callable
        

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

    # def get_timeunit_period(self):
    #     return TimeUnitPeriod(self)

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
            return pd.Timedelta(days=self.quantity * 30)
        elif self.time_unit == 'Y':  # года
            return pd.Timedelta(days=self.quantity * 365)
        

class TimeUnitPeriod:
    def __init__(self, time_unit: TimeUnits):
        self.time_unit = time_unit.time_unit
        self.period_alias = self.get_period_alias()
        self.period_russian_alias = self.get_period_russian_alias()
        self.period_compute_func = self.get_period_compute_func()

    def get_period_alias(self) -> str:
        alias_mapping = {
            "Y": "Year",
            "M": "Month",
            "W": "Week",
            "D": "Day",
            "h": "Hour",
            "m": "Minute",
            "s": "Second",
            "ms": "Millisecond"
        }
        return alias_mapping.get(self.time_unit, "Unknown")

    def get_period_russian_alias(self) -> str:
        russian_alias_mapping = {
            "Y": "Год",
            "M": "Месяц",
            "W": "Неделя",
            "D": "День",
            "h": "Час",
            "m": "Минуты",
            "s": "Секунда",
            "ms": "Миллисекунда"
        }
        return russian_alias_mapping.get(self.time_unit, "Неизвестно")

    def get_period_compute_func(self) -> Callable:
        def compute(data: pd.DataFrame, dt_col: str) -> pd.DataFrame:
            data = data.copy()

            if self.time_unit == "Y":
                data["TimeUnitPeriod"] = data[dt_col].dt.year + self.time_unit.quantity
            elif self.time_unit == "M":
                data["TimeUnitPeriod"] = data[dt_col].dt.month + self.time_unit.quantity
            elif self.time_unit == "W":
                data["TimeUnitPeriod"] = (data[dt_col] + pd.to_timedelta(self.time_unit.quantity * 7, unit='d')).dt.date
            elif self.time_unit == "D":
                data["TimeUnitPeriod"] = (data[dt_col] + pd.to_timedelta(self.time_unit.quantity, unit='d')).dt.date
            elif self.time_unit == "h":
                data["TimeUnitPeriod"] = (data[dt_col] + pd.to_timedelta(self.time_unit.quantity, unit='h'))\
                    .dt.strftime('%Y-%m-%d %H:%M:%S')
            elif self.time_unit == "m":
                data["TimeUnitPeriod"] = (data[dt_col] + pd.to_timedelta(self.time_unit.quantity, unit='m'))\
                    .dt.strftime('%Y-%m-%d %H:%M:%S')
            elif self.time_unit == "s":
                data["TimeUnitPeriod"] = (data[dt_col] + pd.to_timedelta(self.time_unit.quantity, unit='s'))\
                    .dt.strftime('%Y-%m-%d %H:%M:%S')
            elif self.time_unit == "ms":
                data["TimeUnitPeriod"] = (data[dt_col] + pd.to_timedelta(self.time_unit.quantity * 0.001, unit='s'))\
                    .dt.strftime('%Y-%m-%d %H:%M:%S')

            return data

        return compute