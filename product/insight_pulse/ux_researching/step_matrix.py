import itertools
import pandas as pd
from typing import Union, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from eventframing.eventframe import EventFrame
from eventframing.cols_schema import EventFrameColsSchema


pd.set_option('mode.chained_assignment', None)


class StepMatrix:
    # TODO : MOve _path_end_event_name in the bottom of matrix
    _path_end_event_name = 'ENDED'
    _target_cmaps = itertools.cycle(["BrBG", "PuOr", "PRGn", "RdBu"])

    def __init__(self, ef: EventFrame):
        self.ef = ef
        self.cols_schema = ef.cols_schema

    def plot(self, data: Optional[EventFrame] = None, max_steps: int = 10, threshold: float = 0.05,
             weight_col: str = '',
             target_events: List[str] = None, title: str = '') -> None:
        step_matrix = self._fit(data=data, max_steps=max_steps, weight_col=weight_col)
        step_matrix, rare_events = self._threshold_events(step_matrix=step_matrix, threshold=threshold)

        self._plot(step_matrix, rare_events, max_steps=max_steps, title=title, target_events=target_events)

    def plot_difference(self, segment1, segment2, data: Optional[EventFrame] = None, max_steps: int = 10,
                        threshold: float = 0.05, weight_col: str = '', target_events: List[str] = None,
                        title: str = '') -> None:
        difference_matrix = self._fit_matrix_difference(
            segment1,
            segment2,
            data=data,
            max_steps=max_steps,
            weight_col=weight_col
        )

        difference_matrix, rare_events = self._threshold_events(difference_matrix, threshold)
        self._plot(difference_matrix, rare_events, max_steps=max_steps, title=title, target_events=target_events)

    def fit(self, data: Optional[EventFrame] = None, max_steps: int = 10, threshold: float = 0.05,
            weight_col: str = '') -> pd.DataFrame:
        step_matrix = self._fit(data=data, max_steps=max_steps, weight_col=weight_col)
        thresholded_step_matrix, _ = self._threshold_events(step_matrix, threshold)
        return thresholded_step_matrix

    def fit_difference(self, segment1, segment2, data: Optional[EventFrame] = None, max_steps: int = 10,
                       threshold: float = 0.05, weight_col: str = '') -> pd.DataFrame:
        difference_matrix = self._fit_matrix_difference(segment1, segment2, data=data, max_steps=max_steps,
                                                        weight_col=weight_col)
        thresholded_difference_matrix, _ = self._threshold_events(difference_matrix, threshold)
        return thresholded_difference_matrix

    def _get_data_and_schema(self, data: Optional[Union[EventFrame, pd.DataFrame]] = None,
                             cols_schema: Optional[EventFrameColsSchema] = None) -> Tuple[pd.DataFrame, EventFrameColsSchema]:
        if data is None:
            data = self.ef.to_dataframe().copy()
            cols_schema = self.cols_schema
        else:
            if isinstance(data, EventFrame):
                cols_schema = data.cols_schema
                data = data.to_dataframe().copy()
            else:
                cols_schema = cols_schema
                data = data.copy()
        return data, cols_schema

    def _fit_matrix_difference(self, segment1, segment2, data: Optional[EventFrame], max_steps: int,
                               weight_col: str) -> pd.DataFrame:
        data, cols_schema = self._get_data_and_schema(data=data)
        user_col = cols_schema.user_id

        segment1_data = data[data[user_col].isin(segment1)]
        segment2_data = data[data[user_col].isin(segment2)]

        step_matrix1 = self._fit(segment1_data, cols_schema, max_steps=max_steps, weight_col=weight_col)
        step_matrix2 = self._fit(segment2_data, cols_schema, max_steps=max_steps, weight_col=weight_col)

        merged_matrix = pd.merge(
            step_matrix1,
            step_matrix2,
            left_index=True,
            right_index=True,
            suffixes=('_1', '_2'),
            how='outer'
        ).fillna(0)

        difference_matrix = merged_matrix.filter(like='_1').subtract(merged_matrix.filter(like='_2').values)
        difference_matrix.columns = [col.replace('_1', '') for col in difference_matrix.columns]
        return difference_matrix

    def _fit(self, data: Optional[pd.DataFrame] = None, cols_schema: Optional[EventFrameColsSchema] = None,
             max_steps: int = 10, weight_col: str = ''):
        data, cols_schema = self._get_data_and_schema(data=data, cols_schema=cols_schema)

        user_col = cols_schema.user_id
        event_col = cols_schema._event_name
        session_col = cols_schema.session_id
        dt_col = cols_schema.event_timestamp

        if weight_col != '':
            if weight_col not in data.columns:
                raise ValueError(f'Column {weight_col} is not in the EventFrame.')
        else:
            weight_col = session_col if session_col is not None else user_col

        # Add column with number of step in session or the whole path
        data = data.sort_values(by=[user_col, dt_col])
        data['step'] = data.groupby(weight_col)[dt_col].cumcount() + 1
        data = data[data['step'] <= max_steps]

        # Add path terminating event if user has less steps than max_steps
        data = data.pivot_table(index=weight_col, columns='step', values=event_col, aggfunc=lambda x: x) \
            .fillna(self._path_end_event_name).reset_index()

        # Unpivot table into original format but with terminating event
        data = data.melt(id_vars=weight_col, var_name='step', value_name=event_col)

        # Calculate the amount of weight_col units in each step
        data = data.pivot_table(index=event_col, columns='step', values=weight_col, aggfunc='nunique') \
            .fillna(0).sort_values(by=1, ascending=False).astype(float)

        # Normalizing table by columns
        data = data.divide(data.sum()).mul(100).round(1)
        return data

    def _threshold_events(self, step_matrix: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        normal_events = step_matrix.loc[step_matrix.abs().max(axis=1) >= threshold * 100]
        rare_events = step_matrix.loc[step_matrix.abs().max(axis=1) < threshold * 100]

        if not rare_events.empty:
            # create new index respect to amount if events in  thresholded_events
            thresholded_index = f'Thresholded_{rare_events.shape[0]}'
            # Add new string normal_events
            normal_events.loc[thresholded_index] = rare_events.sum()

        return normal_events, rare_events

    def _plot(self, step_matrix: pd.DataFrame, rare_events: pd.DataFrame, max_steps: int, target_events: List[str],
              title: str):
        if target_events is not None:
            target_events = [event for event in target_events
                             if event in step_matrix.index or event in rare_events.index]
            target_events_count = len(target_events)
        else:
            target_events_count = 0
        n_cols = 1
        n_rows = 1 + target_events_count
        fig_hight = (step_matrix.shape[0] + target_events_count) * 0.4
        fig_width = max_steps * 0.6 + 1

        grid_specs = (
            {"wspace": 0.08, "hspace": 0.04,
             "height_ratios": [step_matrix.shape[0], *[1 for _ in range(target_events_count)]]}
            if target_events_count > 0
            else {}
        )
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_hight), gridspec_kw=grid_specs)

        main_matrix_axes = axs[0] if isinstance(axs, np.ndarray) else axs
        sns.heatmap(step_matrix, cmap='Greys', ax=main_matrix_axes, fmt='0.2f', cbar=False, annot=True)
        if not title:
            title = 'Step Matrix'
        main_matrix_axes.set_title(title)

        main_matrix_axes.xaxis.set_ticks_position('top')
        main_matrix_axes.xaxis.tick_top()
        main_matrix_axes.set_xlabel('')

        target_height = fig_hight / (step_matrix.shape[0] + target_events_count)
        if target_events is not None:
            for idx, event in enumerate(target_events):
                target_event_row = step_matrix.loc[[event]] if event in step_matrix.index else rare_events.loc[[event]]
                sns.heatmap(target_event_row, ax=axs[idx + 1], fmt='0.2f', cbar=False, annot=True,
                            cmap=next(self._target_cmaps))

                axs[idx + 1].set_title('')  # Delete title
                axs[idx + 1].xaxis.set_visible(False)
                axs[idx + 1].set_ylabel('')
                axs[idx + 1].set_yticklabels(axs[idx + 1].get_yticklabels(), rotation=0)
