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
    
    @staticmethod
    def _get_data_and_cols_schema(data: Union[pd.DataFrame, 'EventFrame'],
                            cols_schema: Union[Dict[str, str], 'EventFrameColsSchema'],
                            strict: bool = False) -> Tuple[Union[pd.DataFrame, 'EventFrame'], Union[Dict[str, str], 'EventFrameColsSchema']]:
        if isinstance(data, EventFrame):
            return data.data.copy(), data.cols_schema
        elif cols_schema is None:
            if strict:
                raise ValueError("cols_schema is None")
            else:
                return data.copy(), None
        else:
            return data.copy(), EventFrameColsSchema(cols_schema)
        

class MetricKPI(_Metric):
    def __init__(self, formula: Callable[[Union[pd.DataFrame, EventFrame], Optional[EventFrameColsSchema], dict], float], 
                 name: str, description: str):
        super().__init__(formula, name, description)

    def compute_single_value(self, data: Union[pd.DataFrame, 'EventFrame'], cols_schema: Optional[EventFrameColsSchema] = None, **kwargs) -> float:
        data, cols_schema = super()._get_data_and_cols_schema(data, cols_schema, strict=False)
        print(data)
        return self.formula(data, **kwargs)
    
    def compute_splitted_values(self, data: Union[pd.DataFrame, 'EventFrame'],  hue_cols: Union[str, List[str]], 
                                cols_schema: Optional[EventFrameColsSchema] = None, **kwargs) -> pd.DataFrame:
        data, cols_schema = super()._get_data_and_cols_schema(data, cols_schema, strict=False)

        if len(hue_cols) == 0 or hue_cols is None:
            return self.compute_single_value(data, cols_schema, **kwargs)
            
        combinations = self.get_unique_combinations(data, hue_cols)
        result = []
        for combo in combinations:
            combo_desc = combo.copy()
            combo_desc.update({self.name: self.formula(self.filter_data_frame(data, combo), cols_schema, **kwargs)})
            result.append(combo_desc)
            
        return pd.DataFrame(result)
    

class MetricDinamic(_Metric):
    def __init__(self, formula: Callable[[Union[pd.DataFrame, EventFrame], Optional[EventFrameColsSchema], dict], float], 
                 name: str, description: str = ''):
        super().__init__(formula, name, description)

    def _get_data_pivot_template(self, data: pd.DataFrame, cols_schema: EventFrameColsSchema, 
                                 period: TimeUnitPeriod, hue_cols: List[str]) -> pd.DataFrame:
        dt_col = cols_schema.event_timestamp
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
                cols_schema: Union[Dict[str, str], 'EventFrameColsSchema'] = None, 
                fillna_value: float = 0, **kwargs) -> pd.DataFrame:
        
        data, cols_schema = super()._get_data_and_cols_schema(data, cols_schema)

        if isinstance(period, str):
            period = TimeUnitPeriod(period)
        period_name = period.alias
        dt_col = cols_schema.event_timestamp
        data = period.add_period_col(data, dt_col, new_col_name=period_name)
    

        if  hue_cols is None or len(hue_cols) == 0:
            hue_cols = []
            result = data.groupby(period_name)\
                .apply(lambda data: self.formula(data, cols_schema, **kwargs), include_groups=False)\
                    .reset_index().rename(columns={0: self.name})
            
        else:
            if isinstance(hue_cols, str):
                hue_cols = [hue_cols]
            combinations = self.get_unique_combinations(data, hue_cols)
            result = None
            # result = pd.DataFrame(columns=[period_name] + hue_cols + [self.name])
            # return result
            for combo in combinations:
                print(combo)
                combo_result = self.filter_data_frame(data, combo).groupby(period_name)\
                    .apply(lambda data: self.formula(data, cols_schema, **kwargs),
                           include_groups=False)\
                        .reset_index().rename(columns={0: self.name})
                for col_name, col_value in combo.items():
                    combo_result[col_name] = [col_value] * combo_result.shape[0]
                if result is None:
                    result = combo_result.loc[:, tuple([period_name] + hue_cols + [self.name])]
                else:
                    result = pd.concat([result, combo_result.loc[:, tuple([period_name] + hue_cols + [self.name])]], axis=0)
            # return result.sort_values(period_name)
        pivot_template = self._get_data_pivot_template(data, cols_schema, period, hue_cols)
        # return data_date_range
        result = pd.merge(
            pivot_template,
            result,
            on=hue_cols + [period_name],
            how='left'
        )
        result[self.name] = result[self.name].fillna(fillna_value)
        return result.sort_values([period_name] + hue_cols)