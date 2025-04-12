import pandas as pd
from typing import Literal, Union, List, Optional, Iterable, get_args, Dict, Tuple

from data_preprocessing import DataPreprocessor
from eventframing.cols_schema import EventFrameColsSchema
from eventframing.eventframe import EventFrame
from eventframing.event_type import EventType
from utils.time_unit_period import TimeUnitPeriod


CohortPeriods = Literal['D', 'W', 'M', 'Y']


class AddCohortsPreprocessor(DataPreprocessor):

    def __init__(self, cohort_period: Union[TimeUnitPeriod, str] = 'M'):
        if isinstance(cohort_period, str):
            cohort_period = TimeUnitPeriod(cohort_period)
        if cohort_period.time_unit not in get_args(CohortPeriods):
            raise ValueError(f'Invalid cohort period: {cohort_period}')
            
        self.cohort_period = cohort_period
        # TODO: set cohort_group_col_name
        self.cohort_group_col_name = 'cohort_group'

    def apply(self, data: Union[pd.DataFrame, 'EventFrame'],
              cols_schema: Optional[EventFrameColsSchema] = None, prepare: bool = False) -> 'EventFrame':
        # TODO : set event_name to detect cohorts as param

        super()._check_apply_params(data, cols_schema)
        data, cols_schema = super()._get_data_and_cols_schema(data, cols_schema, prepare)

        data = self._delete_synthetic_cols_if_exists(data, cols_schema)
        data = self._extract_cohort(data, cols_schema, self.cohort_period)
        data = self._add_cohort_period(data, cols_schema, self.cohort_period)
        cols_schema.cohort_group = self.cohort_group_col_name

        return EventFrame(data, cols_schema, prepare=False)
    
    def _extract_cohort(self, data: pd.DataFrame,
              cols_schema: EventFrameColsSchema, 
              cohort_period: CohortPeriods = 'month',
              cohort_unit=None) -> pd.DataFrame:

        if cohort_unit is None:
            cohort_unit = cols_schema.user_id
        dt_col = cols_schema.event_timestamp

        cohort_unit_first_appearence = data.groupby(cohort_unit).agg(**{
            'first_action_dt': (dt_col, 'min')
        }).reset_index()

        cohort_unit_first_appearence = cohort_period.add_period_col(
            data=cohort_unit_first_appearence, dt_col='first_action_dt', new_col_name='cohort_group'
        )

        data = cohort_period.add_period_col(
            data=data, dt_col=dt_col, new_col_name='cohort_time_unit'
        )
        
        return pd.merge(
            data,
            cohort_unit_first_appearence.drop(columns=['first_action_dt']),
            on=cohort_unit,
            how='inner'
        )
    
    def _add_cohort_period(self, data: pd.DataFrame,
              cols_schema: EventFrameColsSchema, 
              cohort_period: CohortPeriods = 'month') -> pd.DataFrame:
        data = data.copy()
        data = self.cohort_period.add_period_col(
            data=data, dt_col=cols_schema.event_timestamp, new_col_name='period'
        )

        data['cohort_period'] = self._calculate_period_difference(data, 'cohort_group', 'period', self.cohort_period.time_unit)
        
        return data.drop(columns=['period'])
    
    def _add_period_col(self, data: pd.DataFrame, cohort_period: CohortPeriods, new_col_name: str, dt_col: str) -> pd.DataFrame:
        if cohort_period == 'day':
            data[new_col_name] = pd.to_datetime(data[dt_col].dt.date)
        elif cohort_period == 'week':
            data[new_col_name] = pd.to_datetime(data[dt_col].dt.date) \
                - pd.to_timedelta(data[dt_col].dt.weekday, unit='D')
        elif cohort_period == 'month':
            data[new_col_name] = data[dt_col]\
                .apply(lambda time: time.strftime('%Y-%m'))
        elif cohort_period == 'year':
            data[new_col_name] = data[dt_col].dt.year
        else:
            raise ValueError(f'Unknown cohort period: {cohort_period}')
        return data
    
    def _calculate_period_difference(self, data: pd.DataFrame, col1: str, col2: str, cohort_period: str) -> pd.Series:
        """
        Рассчитывает количество периодов (дней, недель, месяцев или лет) между двумя колонками.

        :param data: DataFrame, содержащий столбцы с периодами.
        :param col1: Название первого столбца.
        :param col2: Название второго столбца.
        :param cohort_period: Период, используемый для вычисления разницы ('day', 'week', 'month', 'year').
        :return: Series с количеством периодов между двумя колонками.
        """
        data = data.copy()
        # data[col1] = pd.to_datetime(data[col1])
        # data[col2] = pd.to_datetime(data[col2]) 

        if cohort_period == 'D':
            return (data[col2] - data[col1]).dt.days  # Разница в днях
        elif cohort_period == 'W':
            return ((data[col2] - data[col1]).dt.days // 7)  # Разница в неделях
        elif cohort_period == 'M':
            return ((data[col2].dt.year - data[col1].dt.year) * 12) + (data[col2].dt.month - data[col1].dt.month)  # Разница в месяцах
        elif cohort_period == 'Y':
            return data[col2].dt.year - data[col1].dt.year  # Разница в годах
        else:
            raise ValueError(f'Unknown cohort period: {cohort_period}')
        
    def _delete_synthetic_cols_if_exists(self, data: pd.DataFrame, cols_schema: EventFrameColsSchema) -> pd.DataFrame:
        cols_to_drop = [self.cohort_group_col_name, 'cohort_time_unit', 'cohort_period']
        cols_to_drop = set(cols_to_drop).intersection(data.columns)
        
        return data.drop(columns=cols_to_drop)
            
