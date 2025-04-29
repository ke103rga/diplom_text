import pandas as pd
from typing import Literal, Union, List, Optional, Iterable, get_args, Dict, Tuple, Callable
from itertools import product
from abc import ABC, abstractmethod
import numpy as np


from eventframing.eventframe import EventFrame
from eventframing.cols_schema import EventFrameColsSchema
from eventframing.event_type import EventType
from utils.time_unit_period import TimeUnitPeriod


class _Metric(ABC):
    def __init__(self, formula: Callable, name: str, description: str = None):
        self.formula = formula
        self.name = name
        self.description = description

    @staticmethod    
    def get_unique_combinations(data: pd.DataFrame, hue_cols: Union[str, List[str]]) -> List[Dict]:
        """
        Возвращает список всех комбинаций уникальных значений полей hue_cols в наборе данных data.
        
        :param data: pd.DataFrame — входные данные.
        :param hue_cols: Union[str, List[str]] — имя колонки или список имен колонок.
        :return: List[Dict] — список комбинаций уникальных значений полей.
        """
        # Если hue_cols - это строка, преобразуем его в список
        if isinstance(hue_cols, str):
            hue_cols = [hue_cols]

        # Проверим, что все столбцы существуют в DataFrame
        for col in hue_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

        # Получаем уникальные значения для каждого из hue_cols
        unique_values = [data[col].unique() for col in hue_cols]

        # Генерируем все комбинации уникальных значений
        combinations = list(product(*unique_values))

        # Создаем результат в виде списка словарей
        result = [{hue_cols[i]: combo[i] for i in range(len(hue_cols))} for combo in combinations]

        return result
    
    @staticmethod    
    def filter_data_frame( data: pd.DataFrame, hue_cols_combo: Dict) -> pd.DataFrame:
        """
        Фильтрует DataFrame по комбинациям уникальных значений hue_cols.
        
        :param data: pd.DataFrame — входные данные.
        :param hue_cols_combos: List[Dict] — список комбинаций уникальных значений hue_cols.
        :return: pd.DataFrame — отфильтрованный DataFrame.
        """
        query = ''
        for col, col_value in hue_cols_combo.items():
            if isinstance(col_value, str) or isinstance(col_value, np.datetime64) or isinstance(col_value, pd.Timestamp):
                query += f"{col} == '{col_value}' & "
            else:
                query += f"{col} == {col_value} & "
        query = query[:-3]
        return data.query(query)
    
    # @staticmethod
    def _get_data(self, data: Union[pd.DataFrame, 'EventFrame']) -> pd.DataFrame:
        
        if isinstance(data, EventFrame):
            return data.data.copy()
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        raise TypeError("data must be pd.DataFrame or EventFrame")
        

class MetricKPI(_Metric):
    def __init__(self, formula: Callable[[Union[pd.DataFrame, EventFrame], Optional[EventFrameColsSchema], dict], float], 
                 name: str, description: Optional[str] = None):
        super().__init__(formula, name, description)

    def _get_data_pivot_template(self, data: pd.DataFrame, hue_cols: List[str]) -> pd.DataFrame:
        unique_values = [data[col].unique() for col in hue_cols]
            
        # Создаем мультииндекс, представляющий все комбинации
        index = pd.MultiIndex.from_product(unique_values, names=hue_cols)
        
        # Создаем DataFrame с этим мультииндексом
        pivot_template = pd.DataFrame(index=index).reset_index()
        return pivot_template

    def compute_single_value(self, data: Union[pd.DataFrame, 'EventFrame'], 
                             formula_kwargs: Optional[Dict] = None) -> float:
        data = super()._get_data(data)
        if formula_kwargs is None:
            formula_kwargs = dict()
        return self.formula(data, **formula_kwargs)
    
    def compute_splitted_values(self, data: Union[pd.DataFrame, 'EventFrame'],  
                                hue_cols: Optional[Union[str, List[str]]] = None, 
                                formula_kwargs: Optional[Dict] = None,
                                fillna_value: float = 0) -> pd.DataFrame:
        data = super()._get_data(data)
        if formula_kwargs is None:
            formula_kwargs = dict()

        if hue_cols is None or len(hue_cols) == 0:
            return self.compute_single_value(data, formula_kwargs)
        
        if isinstance(hue_cols, str):
            hue_cols = [hue_cols]

        result = data.groupby(hue_cols)\
            .apply(lambda group_data: self.formula(group_data, **formula_kwargs), include_groups=False)\
            .reset_index().rename(columns={0: self.name})
        
        pivot_template = self._get_data_pivot_template(data, hue_cols)
    
        # return data_date_range
        result = pd.merge(
            pivot_template,
            result,
            on=hue_cols,
            how='left'
        )
        result[self.name] = result[self.name].fillna(fillna_value)

        return result
    

class MetricDinamicComputeParams():
    def __init__(self, data: Union[pd.DataFrame, 'EventFrame'],
                period: Union[str, TimeUnitPeriod],
                hue_cols: Union[str, List[str]], 
                dt_col: str,
                fillna_value: float, 
                formula_kwargs: Optional[Dict]) -> None:
        
        self.data, self.dt_col = self._get_data_and_dt_col(data, dt_col)

        if isinstance(period, str):
            period = TimeUnitPeriod(period)
        if formula_kwargs is None:
            formula_kwargs = dict()

        self.period = period
        self.period_name = period.alias
        self.hue_cols = hue_cols
        self.fillna_value = fillna_value
        self.formula_kwargs = formula_kwargs
        
    def _get_data_and_dt_col(self,
                            data: Union[pd.DataFrame, 'EventFrame'],
                            dt_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
        
        if isinstance(data, EventFrame):
            dt_col = data.cols_schema.event_timestamp
            data = data.data.copy()
        elif isinstance(data, pd.DataFrame):
            if dt_col is None:
                raise ValueError('It\'s necessary to specify "dt_col" if data is pd.DataFrame')
        else:
            raise TypeError("data must be pd.DataFrame or EventFrame")
        
        return data, dt_col
    

class MetricDinamic(_Metric):
    def __init__(self, formula: Callable[[Union[pd.DataFrame, EventFrame], Optional[EventFrameColsSchema], dict], float], 
                 name: str, description: str = ''):
        super().__init__(formula, name, description)
        self.compute_params: Optional[MetricDinamicComputeParams] = None            

    def _get_data_pivot_template(self, data: pd.DataFrame, dt_col: str, 
                                 period: TimeUnitPeriod, hue_cols: List[str]) -> pd.DataFrame:
        # dt_col = cols_schema.event_timestamp
        min_date, max_date = data[dt_col].min(), data[dt_col].max()
        pivot_template = period.generte_monotic_time_range(min_date, max_date)
        if len(hue_cols) > 0:
            for col_name in hue_cols:
                col_values = data[col_name].unique()
                pivot_template = pd.merge(
                    pivot_template,
                    pd.DataFrame({col_name: col_values}),
                    how='cross'
                )
        return pivot_template

    def compute(self, data: Union[pd.DataFrame, 'EventFrame'],
                period: Union[str, TimeUnitPeriod] = 'D',
                hue_cols: Union[str, List[str]] = None, 
                dt_col: str = None,
                fillna_value: float = 0, 
                formula_kwargs: Optional[Dict] = None) -> pd.DataFrame:
        self.compute_params = MetricDinamicComputeParams(data, period, hue_cols, dt_col, fillna_value, formula_kwargs)
        
        data = self.compute_params.data
        dt_col = self.compute_params.dt_col
        period = self.compute_params.period
        period_name = self.compute_params.period_name
        formula_kwargs = self.compute_params.formula_kwargs
        
        data = period.add_period_col(data, dt_col, new_col_name=period_name)

        if hue_cols is None or len(hue_cols) == 0:
            hue_cols = []
        elif isinstance(hue_cols, str):
            hue_cols = [hue_cols]
        result = data.groupby([period_name] + hue_cols)\
            .apply(lambda data: self.formula(data, **formula_kwargs), include_groups=False)\
            .reset_index().rename(columns={0: self.name})
        
        pivot_template = self._get_data_pivot_template(data, dt_col, period, hue_cols)
    
        # return data_date_range
        result = pd.merge(
            pivot_template,
            result,
            on=hue_cols + [period_name],
            how='left'
        )
        result[self.name] = result[self.name].fillna(fillna_value)
        return result.sort_values([period_name] + hue_cols)