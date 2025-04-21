import pandas as pd
from typing import Optional, Union, List, Tuple

from eventframing.eventframe import EventFrame
from ux_researching.step_matrix import StepMatrix
from ux_researching.step_sankey import StepSankey
from ux_researching.sequences import Sequences
from ux_researching.descriptive_stats import DescStatsAnalyzer
from utils import TimeUnitPeriod, TimeUnits


class UXResearcher:
    def __init__(self, ef: EventFrame):
        self.sequences = Sequences(ef)
        self.step_matrix = StepMatrix(ef)
        self.step_sankey = StepSankey(ef)
        self.desc_stats_analyzer = DescStatsAnalyzer()

    def select_freq_sets(self, data: Optional[EventFrame] = None, ngram_range: Tuple[int, int] = (2, 3),
                         support_threshold: float = 0.05, inside_session: bool = True):
        return self.sequences.select_freq_sets(data, ngram_range, support_threshold, inside_session)

    def associative_rules(self, target_events: Optional[Union[str, List[str]]] = None):
        return self.sequences.associative_rules(target_events)

    def plot_step_matrix(self, data: Optional[EventFrame] = None, max_steps: int = 10, threshold: float = 0.05,
                         weight_col: str = '', target_events: Optional[List[str]] = None, title: str = ''):
        return self.step_matrix.plot(data, max_steps, threshold, weight_col, target_events, title)

    def plot_step_matrix_difference(self, segment1, segment2, data: Optional[EventFrame] = None,
                                    max_steps: int = 10, threshold: float = 0.05, weight_col: str = '',
                                    target_events: Optional[List[str]] = None, title: str = ''):
        return self.step_matrix.plot_difference(segment1, segment2, data, max_steps,
                                                threshold, weight_col, target_events, title)

    def fit_step_matrix(self, data: Optional[EventFrame] = None, max_steps: int = 10, threshold: float = 0.05,
                        weight_col: str = ''):
        return self.step_matrix.fit(data, max_steps, threshold, weight_col)

    def fit_step_matrix_difference(self, segment1, segment2, data: Optional[EventFrame] = None,
                                   max_steps: int = 10, threshold: float = 0.05, weight_col: str = ''):
        return self.step_matrix.fit_difference(segment1, segment2, data, max_steps, threshold, weight_col)

    def fit_sankey(self, data: Optional[EventFrame] = None, max_steps: int = 10, threshold: float = 0.05,
                   weight_col: str = '', events_to_keep: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.step_sankey.fit(data, max_steps, threshold, weight_col, events_to_keep)

    def plot_sankey(self, data: Optional[EventFrame] = None, max_steps: int = 10, threshold: float = 0.05,
                    weight_col: str = '', events_to_keep: Optional[List[str]] = None, title: str = 'StepSankey'):
        return self.step_sankey.plot(data, max_steps, threshold, weight_col, events_to_keep, title)

    def describe(self, ef: EventFrame, add_path_stats: bool = True, add_session_stats: bool = True) -> pd.DataFrame:
        return self.desc_stats_analyzer.describe(ef, add_path_stats, add_session_stats)

    def describe_events(self, ef: EventFrame,
                        events: Optional[List[str]] = None,
                        add_session_stats: bool = True) -> pd.DataFrame:
        return self.desc_stats_analyzer.describe_events(ef, events, add_session_stats)

    def plot_lifetime_hist(self, ef: EventFrame, max_return_time: Optional[Union[int, TimeUnits, Tuple[int, str]]],
                           plot_period: Union[str, TimeUnitPeriod] = 'D', lower_cutoff_quantile: Optional[float] = None,
                           upper_cutoff_quantile: Optional[float] = None, **hist_kwargs) -> None:
        return self.desc_stats_analyzer\
            .plot_lifetime_hist(ef, max_return_time, plot_period, lower_cutoff_quantile,
                                upper_cutoff_quantile, **hist_kwargs)

    def plot_event_distance_hist(self, ef: EventFrame, event_from: Union[str, List[str]], event_to: Union[str, List[str]],
                                 plot_period: Union[str, TimeUnitPeriod] = 'D', lower_cutoff_quantile: Optional[float] = None,
                                 upper_cutoff_quantile: Optional[float] = None, **hist_kwargs) -> None:
        return self.desc_stats_analyzer\
            .plot_event_distance_hist(ef, event_from, event_to, plot_period, lower_cutoff_quantile,
                                      upper_cutoff_quantile, **hist_kwargs)