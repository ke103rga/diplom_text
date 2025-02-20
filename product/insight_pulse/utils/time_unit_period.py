import pandas as pd
# from retentioneering import datasets
from typing import Literal, Union, List, Optional, Iterable, get_args, Dict, Tuple, Callable


class TimeUnitPeriod:
    def __init__(self, time_unit: str):
        self.time_unit = time_unit
        self.alias = self.get_period_alias()
        self.russian_alias = self.get_period_russian_alias()
        # self.period_compute_func = self.get_period_compute_func()

    def get_period_alias(self) -> str:
        alias_mapping = {
            "Y": "Year",
            "M": "Month",
            "W": "Week",
            "D": "date",
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

    def add_period_col(self, data: pd.DataFrame, dt_col: str, new_col_name: Optional[str] = None) -> pd.DataFrame:
        data = data.copy()
        if new_col_name is None or new_col_name == '':
            new_col_name = 'time_unit_period'
        
        if self.time_unit == 'D':
            data[new_col_name] = data[dt_col].dt.date
        elif self.time_unit == 'W':
            data[new_col_name] = (data[dt_col] - pd.to_timedelta(data[dt_col].dt.weekday, unit='D')).dt.date
        elif self.time_unit == 'M':
            data[new_col_name] = data[dt_col].dt.to_period('M').dt.start_time
        elif self.time_unit == 'Y':
            data[new_col_name] = data[dt_col].dt.to_period('Y').dt.start_time
        elif self.time_unit == 'h':
            data[new_col_name] = data[dt_col].dt.floor('H')  # Округляем до часа
        elif self.time_unit == 'm':
            data[new_col_name] = data[dt_col].dt.floor('min')  # Округляем до минуты
        elif self.time_unit == 's':
            data[new_col_name] = data[dt_col].dt.floor('s')  # Округляем до секунды
        elif self.time_unit == 'ms':
            data[new_col_name] = data[dt_col].dt.floor('L')  # Округляем до миллисекунды
        else:
            raise ValueError(f'Unsupported time unit: {self.time_unit}')
        
        # if self.time_unit not in ('Y', 'M'):
        data[new_col_name] = pd.to_datetime(data[new_col_name])
        
        return data
    
    def generte_monotic_time_range(self, min_date: str, max_date: str) -> pd.Series:
        """
        Генерирует непрерывную последовательность pd.Timestamp в зависимости от self.time_unit.

        :param min_date: str — минимальная дата в формате 'YYYY-MM-DD'.
        :param max_date: str — максимальная дата в формате 'YYYY-MM-DD'.
        :return: pd.Series — последовательность дат.
        """
        
        if self.time_unit == 'Y':
            min_date, max_date = pd.Series([min_date, max_date]).dt.floor('Y')
            return pd.date_range(start=min_date, end=max_date, freq='YS').to_series().dt.year.to_frame().rename(columns={0: self.alias})
        elif self.time_unit == 'M':
            min_date, max_date = pd.Series([min_date, max_date]).apply(lambda time: time.strftime('%Y-%m'))
            return pd.date_range(start=min_date, end=max_date, freq='MS').to_series().apply(lambda time: time.strftime('%Y-%m')).to_frame().rename(columns={0: self.alias})
        elif self.time_unit == 'W':
            range_dates = pd.Series([min_date, max_date])
            range_dates = pd.to_datetime(range_dates.dt.date) - pd.to_timedelta(range_dates.dt.weekday, unit='D')
            min_date, max_date = range_dates
            return pd.date_range(start=min_date, end=max_date, freq='W-MON').to_frame().rename(columns={0: self.alias})
        elif self.time_unit == 'D':
            min_date, max_date = pd.Series([min_date, max_date]).dt.floor('D')
            return pd.date_range(start=min_date, end=max_date, freq='D').to_frame().rename(columns={0: self.alias})
        elif self.time_unit == 'h':
            min_date, max_date = pd.Series([min_date, max_date]).dt.floor('h')
            return pd.date_range(start=min_date, end=max_date, freq='h').to_frame().rename(columns={0: self.alias})
        elif self.time_unit == 'm':
            min_date, max_date = pd.Series([min_date, max_date]).dt.floor('min')
            return pd.date_range(start=min_date, end=max_date, freq='min').to_frame().rename(columns={0: self.alias})
        elif self.time_unit == 's':
            min_date, max_date = pd.Series([min_date, max_date]).dt.floor('s')
            return pd.date_range(start=min_date, end=max_date, freq='s').to_frame().rename(columns={0: self.alias})
        elif self.time_unit == 'ms':
            min_date, max_date = pd.Series([min_date, max_date]).dt.floor('L')
            return pd.date_range(start=min_date, end=max_date, freq='L').to_frame().rename(columns={0: self.alias})
        else:
            raise ValueError(f'Unsupported time unit: {self.time_unit}')