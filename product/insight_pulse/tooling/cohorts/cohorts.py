from typing import Literal, Union, List, Optional, Iterable, get_args
import pandas as pd
import numpy as np
import seaborn as sns


from eventframing.cols_schema import EventFrameColsSchema
from eventframing.eventframe import EventFrame
from metrics.metric import MetricKPI
from data_preprocessing.preprocessors_lib.add_cohorts_preprocessor import AddCohortsPreprocessor
from utils.time_unit_period import TimeUnitPeriod


class Cohorts:
    """
    A class for working with cohorts.
    """ 
    RepresentationTypes = Literal['time_unit', 'period']

    def __init__(self):
        self.kpi_metric = None
        self.cohort_period = None
        self.represent_by = None
        self.cohort_table = None
        self.normalize = False
        

    def fit(self, data: Union[pd.DataFrame, 'EventFrame'],
            cols_schema: Optional[EventFrameColsSchema],
            kpi_metric: MetricKPI,
            extract_cohorts: bool = True,
            cohort_period: Union[TimeUnitPeriod, str] = 'D',
            represent_by:RepresentationTypes ='time_unit',
            normalize: bool = False) -> pd.DataFrame:
        self._check_fit_params(data, cols_schema, represent_by)

        if isinstance(data, EventFrame):
            data = data.data.copy()
            cols_schema = data.cols_schema
        else:
            data = data.copy()

        if isinstance(cohort_period , str):
            cohort_period = TimeUnitPeriod(cohort_period)

        if extract_cohorts:
            cohorts_preprocessor = AddCohortsPreprocessor(cohort_period)
            data = cohorts_preprocessor.apply(data, cols_schema)

        represent_by_col = 'cohort_time_unit' if represent_by == 'time_unit' else 'cohort_period'

        self.cohort_period = cohort_period
        self.represent_by = represent_by
        self.kpi_metric = kpi_metric
        
        cohort_table = kpi_metric.compute_splitted_values(
            data=data, 
            cols_schema=cols_schema, 
            hue_cols=['cohort_group', represent_by_col]
        )

        pivot_template = self._prepare_pivot_template(cohort_table, represent_by)
        
        cohort_table = pd.merge(
            pivot_template,
            cohort_table,
            how='left',
            on=['cohort_group', represent_by_col]
        ).fillna(0)
        
        cohort_table = cohort_table.pivot_table(
            index='cohort_group', 
            columns=represent_by_col,
            values=kpi_metric.name,
            aggfunc=lambda x: x
        )

        if normalize:
            if represent_by == 'time_unit':
                cohort_table = cohort_table.divide(np.diag(cohort_table), axis=0).mul(100)
            else:
                cohort_table = cohort_table.divide(cohort_table.iloc[:, 0], axis=0).mul(100)

        self.cohort_table = cohort_table
        return cohort_table
    
    def plot(self, annot: bool = True, fmt=None, annot_kws=None, cmap=None, title=None,
             min_period: Optional[int] = None, max_period: Optional[int] = None, 
             min_time_unit: Optional[Union[str, pd.Timestamp]] = None, 
             max_time_unit: Optional[Union[str, pd.Timestamp]] = None) -> None:
        
        cohort_table = self.cohort_table.copy()

        # here

        cohort_table.set_index(cohort_table.index.astype(str), inplace=True)
        cohort_table.columns = cohort_table.columns.astype(str)
            
        if fmt is None:
            fmt = '.0%' if self.normalize else '.2f'
        if annot_kws is None:
            annot_kws = {'fontsize': 10}
        if cmap is None:
            cmap = sns.color_palette("light:b", as_cmap=True)
        if title is None:
            title = f'{self.kpi_metric.name} by {self.cohort_period.alias} cohorts'

        if self.represent_by == 'time_unit':
            ylabel = self.cohort_period.alias
        else:
            ylabel = f'{self.cohort_period.alias}(s) after first visit'
        
        
        fig, axes = plt.subplots(figsize=(12, 6))
        sns.heatmap(data=cohort_table, mask=cohort_table.isnull(),
                    annot=annot, fmt=fmt, ax=axes, annot_kws=annot_kws, cmap=cmap)
        axes.set_title(title)
        axes.set_xlabel(ylabel)
        axes.set_ylabel("Cohort group")
        plt.tight_layout()

    @property
    def values(self):
        return self.cohort_table.copy()
    
    def _prepare_cohort_table(self, min_period: Optional[int] = None, max_period: Optional[int] = None, 
            min_time_unit: Optional[Union[str, pd.Timestamp]] = None, 
            max_time_unit: Optional[Union[str, pd.Timestamp]] = None,
            group_mean: bool = False, period_mean: bool = False) -> pd.DataFrame:
        
        cohort_table = self.cohort_table.copy()

        if self.represent_by == 'time_unit':
            min_time_unit = np.datetime64(min_time_unit) if min_time_unit is not None else min(cohort_table.columns)
            max_time_unit = np.datetime64(max_time_unit) if max_time_unit is not None else max(cohort_table.columns)
            cols = (np.datetime64(col) for col in cohort_table.columns if col >= min_time_unit and col <= max_time_unit)
            cols = tuple(cols)
            if (len(list(cols)) == 0):
                raise ValueError(f'No cohort periods which are between {str(min_time_unit)} and {str(max_time_unit)}')
        if self.represent_by == 'period':
            min_period = int(min_period) if min_period is not None else min(cohort_table.columns)
            max_period = int(max_period) if max_period is not None else max(cohort_table.columns)
            cols = (col for col in cohort_table.columns if col >= min_period and col <= max_period)
            cols = tuple(cols)
            
            if (len(list(cols)) == 0):
                raise ValueError(f'No cohort periods which are between {str(min_period)} and {str(max_period)}')
        cohort_table = cohort_table.loc[:, tuple(cols)]

        if group_mean:
            cohort_table['cohort_group_mean'] = cohort_table.mean(axis=1)
        if period_mean:
            cohort_table.loc['period_mean'] = cohort_table.mean(axis=0)

        return cohort_table

    def _prepare_pivot_template(self, cohorts_data: pd.DataFrame, represent_by: str) -> pd.DataFrame:
        time_unit_name = self.cohort_period.alias
        cohort_group_min, cohort_group_max =  cohorts_data['cohort_group'].min(), cohorts_data['cohort_group'].max()
        cohort_group_monotic_range = self.cohort_period.generte_monotic_time_range(cohort_group_min, cohort_group_max)\
                .rename(columns={time_unit_name: 'cohort_group'})
        
        if represent_by == 'time_unit':
            cohort_tu_min, cohort_tu_max =  cohorts_data['cohort_time_unit'].min(), cohorts_data['cohort_time_unit'].max()
            cohort_tu_monotic_range = self.cohort_period.generte_monotic_time_range(cohort_tu_min, cohort_tu_max)\
                    .rename(columns={time_unit_name: 'cohort_time_unit'})
            
            pivot_template = pd.merge(
                cohort_group_monotic_range,
                cohort_tu_monotic_range,
                how='cross'
            )
            pivot_template = pivot_template[pivot_template['cohort_time_unit'] >= pivot_template['cohort_group']]
                    
        else:
            cohort_per_min, cohort_per_max =  cohorts_data['cohort_period'].min(), cohorts_data['cohort_period'].max()
            cohort_per_monotic_range = pd.Series(list(range(cohort_per_min, cohort_per_max + 1))).to_frame(name='cohort_period')
            pivot_template = pd.merge(
                cohort_group_monotic_range,
                cohort_per_monotic_range,
                how='cross'
            )

        return pivot_template        
            

    def _check_fit_params(
            self,
            data: Union[pd.DataFrame, 'EventFrame'],
            cols_schema: Optional[EventFrameColsSchema],
            represent_by:RepresentationTypes 
    ):
        if not (isinstance(data, EventFrame) or
                (isinstance(data, pd.DataFrame) and isinstance(cols_schema, EventFrameColsSchema))):
            raise ValueError('EventFrame or DataFrame with EventFrameColsSchema')
        
        if not represent_by in get_args(self.RepresentationTypes):
            raise ValueError(f'Invalid representation type: {represent_by}')
        

