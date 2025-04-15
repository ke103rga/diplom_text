from product.insight_pulse.eventframing.eventframe import EventFrame

import pandas as pd
from typing import Union, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod
import warnings

import datetime


class DescStatsAnalyzer:
    _default_agg_funcs = ['mean', 'median', 'std', 'min', 'max']

    @staticmethod
    def describe(ef: EventFrame, add_path_stats: bool = True, add_session_stats: bool = True) -> pd.DataFrame:
        data = ef.to_dataframe().copy()
        cols_schema = ef.cols_schema

        user_col = cols_schema.user_id
        event_col = cols_schema.event_name
        session_col = cols_schema.session_id
        dt_col = cols_schema.event_timestamp
        event_type_col = cols_schema.event_type

        raw_data = data[data[event_type_col] == EventType.RAW.value.name].copy()

        overall_stats = []
        overall_stats.append(['unique_users', raw_data[user_col].nunique()])
        overall_stats.append(['unique_events', raw_data[event_col].nunique()])
        if session_col:
            overall_stats.append(['unique_sessions', raw_data[session_col].nunique()])
        overall_stats.append(['frame_start', raw_data[dt_col].min()])
        overall_stats.append(['frame_end', raw_data[dt_col].max()])
        overall_stats.append(['frame_duration', raw_data[dt_col].max() - raw_data[dt_col].min()])
        # Add 'overall' in start of every sub list
        for overall_stats_row in overall_stats:
            overall_stats_row.insert(0, 'overall')

        path_stats = []
        if add_path_stats:
            path_stats = raw_data.groupby(user_col).agg(**{
                'path_duration': (dt_col, lambda x: x.max() - x.min()),
                'path_steps_length': (event_col, 'count'),
            })

            path_stats = DescStatsAnalyzer._compute_desc_stats(path_stats)

        session_stats = []
        if add_session_stats:
            if not session_col:
                raise ValueError(
                    'session_id column not defined. It is necessary to split Eventframe by sessions previously')

            session_stats = raw_data.groupby(session_col).agg(**{
                'session_duration': (dt_col, lambda x: x.max() - x.min()),
                'session_steps_length': (event_col, 'count'),
            })

            session_stats = DescStatsAnalyzer._compute_desc_stats(session_stats)

        overall_stats.extend(path_stats)
        overall_stats.extend(session_stats)
        return pd.DataFrame(overall_stats, columns=['category', 'metric', 'value']).set_index(['category', 'metric'])

    @staticmethod
    def describe_events(ef: EventFrame, events: Optional[List[str]] = None,
                        add_session_stats: bool = True) -> pd.DataFrame:
        data = ef.to_dataframe().copy()
        cols_schema = ef.cols_schema

        user_col = cols_schema.user_id
        event_col = cols_schema.event_name
        session_col = cols_schema.session_id
        dt_col = cols_schema.event_timestamp
        event_type_col = cols_schema.event_type

        if add_session_stats and not session_col:
            raise ValueError(
                'session_id column not defined. It is necessary to split Eventframe by sessions previously')

        raw_data = data[data[event_type_col] == EventType.RAW.value.name].copy()
        # Get the first action for every user
        users_first_events = raw_data.groupby(user_col).agg(
            first_path_event_dt=(dt_col, 'min'),
        ).reset_index()
        # Add the step number in user's path
        raw_data['path_step_number'] = raw_data.groupby(user_col).cumcount() + 1

        if events:
            raw_data = raw_data[raw_data[event_col].isin(events)].copy()

        events_users_stats = raw_data.groupby([user_col, event_col]).agg(
            first_occarance_dt=(dt_col, 'min'),
            steps_to_first_occurance=('path_step_number', 'min'),
        ).reset_index()
        events_users_stats = events_users_stats.merge(
            users_first_events,
            on=user_col,
            how='inner'
        )
        events_users_stats['time_to_first_occurrence'] = events_users_stats['first_occurrence_dt'] - events_users_stats[
            'first_path_event_dt']
        events_users_stats = events_users_stats.drop(columns=['first_occurrence_dt'])

        # For time_to_first_occurrence
        time_stats = events_users_stats.groupby(event_col)['time_to_first_occurrence'].agg(
            DescStatsAnalyzer._default_agg_funcs).reset_index()
        time_stats.set_index(event_col, inplace=True)
        time_stats.columns = pd.MultiIndex.from_product(
            [['time_to_first_occurrence'], DescStatsAnalyzer._default_agg_funcs])

        # For steps_to_first_occurrence
        steps_stats = events_users_stats.groupby(event_col)['steps_to_first_occurrence'].agg(
            DescStatsAnalyzer._default_agg_funcs).reset_index()
        steps_stats.set_index(event_col, inplace=True)
        steps_stats.columns = pd.MultiIndex.from_product(
            [['steps_to_first_occurrence'], DescStatsAnalyzer._default_agg_funcs])

        overall_stats_agg = {
            'event_count': (event_col, 'count'),
            'unique_users': (user_col, 'nunique'),
        }
        if add_session_stats and session_col:
            overall_stats_agg['unique_sessions'] = (session_col, 'nunique')

        overall_event_stats = raw_data.groupby(event_col).agg(**overall_stats_agg).reset_index()

        # Compute percents
        total_events = raw_data.shape[0]
        total_users = raw_data[user_col].nunique()
        overall_event_stats['event_percentage'] = (overall_event_stats['event_count'] / total_events * 100).fillna(0)
        overall_event_stats['user_percentage'] = (overall_event_stats['unique_users'] / total_users * 100).fillna(0)

        if add_session_stats and session_col:
            total_sessions = raw_data[session_col].nunique()
            overall_event_stats['session_percentage'] = (
                        overall_event_stats['unique_sessions'] / total_sessions * 100).fillna(0)

        overall_event_stats.set_index(event_col, inplace=True)
        overall_event_stats.columns = pd.MultiIndex.from_product([['overall'], overall_event_stats.columns])

        event_stats = overall_event_stats \
            .merge(time_stats, right_index=True, left_index=True) \
            .merge(steps_stats, right_index=True, left_index=True)

        return event_stats.reset_index()

    @staticmethod
    def plot_lifetime_hist(ef: EventFrame,
                           max_return_time: Optional[Union[int, TimeUnits, Tuple[int, str]]],
                           plot_period: Union[str, TimeUnitPeriod] = 'D',
                           lower_cutoff_quantile: Optional[float] = None,
                           upper_cutoff_quantile: Optional[float] = None,
                           **hist_kwargs) -> None:
        # Обработка max_return_time
        if isinstance(max_return_time, int):
            max_return_time = pd.Timedelta(minutes=max_return_time)
        elif isinstance(max_return_time, tuple):
            max_return_time = TimeUnits(max_return_time).get_time_delta()
        elif isinstance(max_return_time, TimeUnits):
            max_return_time = max_return_time.get_time_delta()
        else:
            raise ValueError('max_return_time should be an int or TimeUnits')

        if isinstance(plot_period, str):
            plot_period = TimeUnitPeriod(plot_period)

        data = ef.to_dataframe().copy()
        cols_schema = ef.cols_schema

        user_col = cols_schema.user_id
        dt_col = cols_schema.event_timestamp

        # Для каждого пользователя вычисляем время первого и последнего действия
        user_life_data = data.groupby(user_col).agg(
            first_action_dt=(dt_col, 'min'),
            last_action_dt=(dt_col, 'max')
        ).reset_index()

        # Исключаем пользователей, чье последнее действие позже max_return_time
        max_dt_in_data = data[dt_col].max()
        user_life_data = user_life_data[user_life_data['last_action_dt'] <= (max_dt_in_data - max_return_time)]

        # Вычисляем продолжительность жизни пользователей
        user_life_data['lifetime'] = user_life_data['last_action_dt'] - user_life_data['first_action_dt']

        lifetimes = DescStatsAnalyzer._prepare_dist_timedelta_data(
            data=user_life_data,
            dist_col='distance',
            new_col_name=f'lifetime ({plot_period.alias})',
            dist_period=plot_period,
            lower_cutoff_quantile=lower_cutoff_quantile,
            upper_cutoff_quantile=upper_cutoff_quantile
        )
        DescStatsAnalyzer._plot_seaborn_hist(
            lifetimes,
            title='Distribution of Event Distance',
            xaxis_label=f'Lifetime ({plot_period.alias})',
            yaxis_label='Users',
            **hist_kwargs
        )

    def plot_event_distance_hist(ef: EventFrame,
                                 event_from: Union[str, List[str]],
                                 event_to: Union[str, List[str]],
                                 plot_period: Union[str, TimeUnitPeriod] = 'D',
                                 lower_cutoff_quantile: Optional[float] = None,
                                 upper_cutoff_quantile: Optional[float] = None,
                                 **hist_kwargs) -> None:
        data = ef.to_dataframe().copy()
        cols_schema = ef.cols_schema

        if isinstance(plot_period, str):
            plot_period = TimeUnitPeriod(plot_period)

        user_col = cols_schema.user_id
        event_col = cols_schema.event_name
        dt_col = cols_schema.event_timestamp

        # Check that all events appear in data
        all_events = data[event_col].unique()
        events_to_check = [event_from] if isinstance(event_from, str) else event_from.copy()
        events_to_check += [event_to] if isinstance(event_to, str) else event_to.copy()
        for event in events_to_check:
            if event not in all_events:
                raise ValueError(f"Event '{event}' not found in the EventFrame.")

        # Renaming events which were passed as list
        if isinstance(event_from, list):
            event_replace = dict(zip(event_from, ['event_from'] * len(event_from)))
            print(event_replace)
            data.event = data.event.replace(event_replace, regex=False)
            event_from = 'event_from'

        if isinstance(event_to, list):
            event_replace = dict(zip(event_to, ['event_to'] * len(event_to)))
            print(event_replace)
            data.event = data.event.replace(event_replace, regex=False)
            event_to = 'event_to'

        data = data[data[event_col].isin([event_from, event_to])]

        data = data.sort_values(by=[user_col, dt_col])
        # Compute distance between current step event timestamp and next step event timestamp
        data['distance'] = data.groupby(user_col)[dt_col].shift(-1) - data[dt_col]
        # Hold only relevant rows
        distance_data = data[(data[event_col] == event_from) & (data[event_col].shift(-1) == event_to)]

        distances = DescStatsAnalyzer._prepare_dist_timedelta_data(
            data=distance_data,
            dist_col='distance',
            new_col_name=f'distance ({plot_period.alias})',
            dist_period=plot_period,
            lower_cutoff_quantile=lower_cutoff_quantile,
            upper_cutoff_quantile=upper_cutoff_quantile
        )
        DescStatsAnalyzer._plot_seaborn_hist(
            distances,
            title='Distribution of Event Distance',
            xaxis_label=f'Distance ({plot_period.alias})',
            yaxis_label='Stat',
            **hist_kwargs
        )

    def _prepare_dist_timedelta_data(
            data: pd.DataFrame,
            dist_col: str,
            new_col_name: str,
            dist_period: Union[str, TimeUnitPeriod],
            lower_cutoff_quantile: Optional[float] = None,
            upper_cutoff_quantile: Optional[float] = None, ) -> pd.Series:
        if isinstance(dist_period, str):
            dist_period = TimeUnitPeriod(dist_period)
        if lower_cutoff_quantile is None:
            lower_cutoff_quantile = 0
        if upper_cutoff_quantile is None:
            upper_cutoff_quantile = 1

        data = dist_period.add_period_col_from_timedelta(data, dist_col, new_col_name)
        data = data[
            (data[new_col_name] >= data[new_col_name].quantile(lower_cutoff_quantile)) &
            (data[new_col_name] <= data[new_col_name].quantile(upper_cutoff_quantile))
            ]
        return data[new_col_name]

    def _plot_seaborn_hist(data: pd.Series, title: str, xaxis_label: str, yaxis_label: str, **hist_kwargs) -> None:
        fig, axes = plt.subplots(figsize=(12, 6))
        sns.histplot(data, kde=True, **hist_kwargs)

        axes.set_title(title)
        axes.set_xlabel(xaxis_label)
        axes.set_ylabel(yaxis_label)
        plt.tight_layout()
        plt.show()

    def _compute_desc_stats(data: pd.DataFrame,
                            agg_cols: Optional[List[str]] = None,
                            agg_funcs: Optional[List[str]] = None) -> List[List]:
        if not agg_cols:
            agg_cols = data.columns
        if not agg_funcs:
            agg_funcs = DescStatsAnalyzer._default_agg_funcs

        stats = []
        for col in agg_cols:
            series = data[col]
            for agg_func in agg_funcs:
                stats.append([col, agg_func, series.apply(agg_func)])

        return stats