from typing import Literal, Union, List, Optional, Iterable, get_args, Tuple
import pandas as pd
from plotly import graph_objects as go
import random

from data_preprocessing.preprocessors_lib.add_start_end_events import AddStartEndEventsPreprocessor
from data_preprocessing.preprocessors_lib.split_sessions import SplitSessionsPreprocessor

from eventframing.cols_schema import EventFrameColsSchema
from eventframing.eventframe import EventFrame

FunnelTypes = Literal["open", "closed"]


class Funnel:
    """
    A class for building product funnels based on an EventFrame.

    The Funnel class allows specifying stages (either as single events or groups of events),
    as well as names for these stages. It also supports segmenting the funnel into an
    unlimited number of segments.

    Attributes:
        _funnel_data (pd.DataFrame): The processed funnel data containing stages and user counts.
        stages (List[str]): A list of funnel stages.
        stages_names (List[str]): A list of names corresponding to the funnel stages.
    """
    _funnel_data: pd.DataFrame
    stages = List[str]
    stages_names = List[str]

    def __init__(self):
        pass

    def _check_fit_params(
            self,
            funnel_type: FunnelTypes,
            data: Union[pd.DataFrame, 'EventFrame'],
            stages: List[Union[str, List[str]]],
            stages_names: Optional[List[str]],
            inside_session: bool,
            segments: Optional[Iterable],
            segments_names: Optional[Iterable]
    ):
        """
        Checks the parameters provided to the fit method for validity.

        Args:
            funnel_type (FunnelTypes): The type of funnel ('open' or 'closed').
            data (Union[pd.DataFrame, EventFrame]): The data to fit the funnel on.
            stages (List[Union[str, List[str]]]): The stages defined for the funnel.
            stages_names (Optional[List[str]]): The names for the stages.
            inside_session (bool): Indicator if the funnel should be calculated inside a session.
            segments (Optional[Iterable]): The segments to analyze.
            segments_names (Optional[Iterable]): The names for the segments.

        Raises:
            ValueError: If any of the parameters are invalid.
        """
        if funnel_type not in get_args(FunnelTypes):
            raise ValueError(f'funnel_type should be one of {get_args(FunnelTypes)}')

        if not isinstance(data, EventFrame):
            raise ValueError('only EventFrame')

        self._check_repeated_stages(stages=stages)

        if stages_names:
            if not len(set(stages_names)) == len(stages_names):
                raise ValueError(f'All stages names should be distinct')

            if not len(stages) == len(stages_names):
                raise ValueError(f'Amount of Stages differ from amount of their names')

        if segments:
            # if self._segments_repeated_indexes(segments):
            #     raise ValueError(f'Segments should not contain repeating elements')

            if segments_names is not None and not len(segments) == len(segments_names):
                raise ValueError(f'Amount of segments differ from amount of their names')

    def _segments_repeated_indexes(self, segments):
        """
        Checks if any indexes are repeated across the provided segments.

        Args:
            segments (Iterable): The segments to check for repeated indexes.

        Returns:
            bool: True if repeated indexes exist, False otherwise.
        """
        segment_idx = set()

        for segment in segments:
            segment_set = set(segment)
            if len(segment_idx.intersection(segment_set)) > 0:
                return False
            segment_idx = segment_idx.union(segment_set)
        return True

    def _check_repeated_stages(self, stages: List[Union[str, List[str]]]) -> None:
        """
        Validates that all stages provided are either strings or lists of strings and distinct.

        Args:
            stages (List[Union[str, List[str]]]): The stages to validate.

        Raises:
            ValueError: If any stages are invalid or repeated.
        """
        multiple_stages = []
        all_stages = []
        for stage in stages:
            if isinstance(stage, list):
                multiple_stages.append(stage)
            elif isinstance(stage, str):
                all_stages.append(stage)
            else:
                raise ValueError(f'Stages should be either str or list')

        for multiple_stage in multiple_stages:
            for stage in multiple_stage:
                if not isinstance(stage, str):
                    raise ValueError(f'Stages should be either str or list of str')
                all_stages.append(stage)

        if not len(set(all_stages)) == len(all_stages):
            raise ValueError(f'All stages should be distinct')

    def _collapse_stages(self, data: pd.DataFrame, cols_schema: EventFrameColsSchema,
                         stages: List[Union[str, List[str]]], stages_names: Optional[List[str]]
                         ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Collapses the stages into unique identifiers and updates the event names in the data.

        Args:
            data (pd.DataFrame): The DataFrame containing events.
            cols_schema (EventFrameColsSchema): The schema of the EventFrame columns.
            stages (List[Union[str, List[str]]]): The stages to collapse.
            stages_names (Optional[List[str]]): The names associated with the stages.

        Returns:
            Tuple[pd.DataFrame, List[str], List[str]]: Updated DataFrame, new stages, and their names.
        """
        new_stages = []
        multiple_stages = []
        new_stages_names = []

        collapse_dict = dict()

        for stage in stages:
            if isinstance(stage, str):
                new_stages.append(stage)
                new_stages_names.append(stage)
            else:
                multiple_stages.append(stage)
                new_stage_name = '__'.join(stage)
                new_stages.append(new_stage_name)
                new_stages_names.append(new_stage_name)
                for sub_stage in stage:
                    collapse_dict[sub_stage] = new_stage_name

        if stages_names is None:
            stages_names = new_stages_names

        data_copy = data.copy()
        event_col = cols_schema.event_name
        data_copy[event_col] = data_copy[event_col].replace(collapse_dict, inplace=False)

        return data_copy, new_stages, stages_names

    def fit(
            self,
            data: EventFrame,
            stages: List[Union[str, List[str]]],
            funnel_type: FunnelTypes = 'open',
            stages_names: Optional[List[str]] = None,
            inside_session: bool = False,
            segments: Optional[Iterable] = None,
            segments_names: Optional[Iterable] = None
    ) -> pd.DataFrame:
        """
        Fits the funnel model to the provided EventFrame data and prepares the funnel data.

        Args:
            data (EventFrame): The event data to process.
            stages (List[Union[str, List[str]]]): The stages defined for the funnel.
            funnel_type (FunnelTypes, optional): The type of funnel to build ('open' or 'closed'). Defaults to 'open'.
            stages_names (Optional[List[str]], optional): The names for the stages. Defaults to None.
            inside_session (bool, optional): Indicator if the funnel should be calculated inside a session. Defaults to False.
            segments (Optional[Iterable], optional): The segments to analyze. List of lists of indexes in data
            to separate different segments of users. Defaults to None.
            segments_names (Optional[Iterable], optional): The names for the segments. Defaults to None.

        Returns:
            pd.DataFrame: The fitted funnel data with stages and user counts.

        Raises:
            ValueError: If any of the parameters are invalid.
        """
        self._check_fit_params(funnel_type, data, stages, stages_names,
                               inside_session, segments, segments_names)

        cols_schema = data.cols_schema
        data = data.data.copy()

        data, stages, stages_names = self._collapse_stages(data, cols_schema, stages, stages_names)
        self.stages = stages
        self.stages_names = stages_names

        user_id_col = cols_schema.user_id
        event_col = cols_schema.event_name

        funnel_data = pd.DataFrame(columns=['stage', 'users_count', 'segment'])

        if segments is None:
            segments, segments_names = [data.index], ['all_users']
        elif segments_names is None:
            segments_names = [f'segment_{i}' for i in range(len(segments))]

        for segment, segment_name in zip(segments, segments_names):
            segment_data = data.loc[segment]
            segment_data = segment_data[segment_data[event_col].isin(stages)]

            if funnel_type == 'open':
                segment_funnel_data = self._fit_open_funnel(segment_data, cols_schema,
                                                            stages=stages, inside_session=inside_session)

            else:
                segment_funnel_data = self._fit_close_funnel(segment_data, cols_schema,
                                                             stages=stages, inside_session=inside_session)

            # Create the dictionary that defines the order for sorting
            sorter_index = dict(zip(stages, range(len(stages))))
            segment_funnel_data['stage_rank'] = segment_funnel_data['stage'].map(sorter_index)
            segment_funnel_data = segment_funnel_data.sort_values(by='stage_rank', ascending=True)
            segment_funnel_data = segment_funnel_data.drop(columns=['stage_rank'])

            segment_funnel_data['segment'] = segment_name

            funnel_data = pd.concat([funnel_data, segment_funnel_data])

        funnel_data['stage'] = funnel_data['stage'].replace(
            {stage: stage_name for stage, stage_name in zip(stages, stages_names)}
        )

        funnel_data['users_count'] = funnel_data['users_count'].astype(int)
        self._funnel_data = funnel_data
        return funnel_data

    def _fit_open_funnel(
            self,
            data: pd.DataFrame,
            cols_schema: EventFrameColsSchema,
            stages: List[str],
            inside_session: bool
    ) -> pd.DataFrame:
        user_id_col = cols_schema.user_id
        event_col = cols_schema.event_name

        if inside_session:
            raise ValueError('Inside session only closed funnel')

        funnel_data = data.pivot_table(
            index=event_col,
            values=user_id_col,
            aggfunc='nunique'
        ).reset_index()

        funnel_data = funnel_data.rename(columns={
            event_col: 'stage',
            user_id_col: 'users_count'
        })

        return funnel_data

    def _fit_close_funnel(
            self,
            data: pd.DataFrame,
            cols_schema: EventFrameColsSchema,
            stages: List[str],
            inside_session: bool
    ) -> pd.DataFrame:
        user_id_col = cols_schema.user_id
        event_col = cols_schema.event_name
        dt_col = cols_schema.event_timestamp
        event_id_col = cols_schema.event_id

        if inside_session:
            # TODO: release logic of calculating inside session funnel
            pass

        else:
            stages_user_counts = []
            resident_data = data.copy()
            for stage in stages:
                first_stage_events = resident_data[resident_data[event_col] == stage]
                if first_stage_events.shape[0] == 0:
                    break
                first_stage_events = first_stage_events.sort_values(by=[user_id_col, dt_col])
                first_stage_events = first_stage_events.groupby(user_id_col).head(1)\
                                     .loc[:, (user_id_col, dt_col, event_id_col)]
                stages_user_counts.append(first_stage_events[user_id_col].nunique())

                resident_data = pd.merge(
                    resident_data,
                    first_stage_events.rename(columns={dt_col: 'first_action_dt', event_id_col: 'first_action_id'}),
                    how='left',
                    on=user_id_col
                ).reset_index()
                resident_data['first_action_dt'] = resident_data['first_action_dt'].fillna(pd.to_datetime('2090-01-01', yearfirst=True))

                resident_data = resident_data[
                    # (resident_data[event_id_col] == resident_data['first_action_id']) |
                    (resident_data[dt_col] >= resident_data['first_action_dt'])
                ].drop(columns=['first_action_dt', 'first_action_id', 'index'])
                if resident_data.shape[0] == 0:
                    break

            stages_user_counts.extend([0] * (len(stages) - len(stages_user_counts)))
            return pd.DataFrame(data={
                'stage': stages,
                'users_count': stages_user_counts
            })

    def calculate_percentage(self, data: pd.DataFrame) -> pd.DataFrame:
        """
       Calculates the percentage of users at each stage of the funnel.

       Args:
           data (pd.DataFrame): The DataFrame containing the funnel data.

       Returns:
           pd.DataFrame: The original DataFrame with additional columns for percentage calculations.
       """
        data['percent_of_previous'] = data['users_count'] / data['users_count'].shift(1) * 100
        data['percent_of_previous'] = data['percent_of_previous'].fillna(100)

        data['percent_of_initial'] = data['users_count'] / data['users_count'].iloc[0] * 100

        return data

    @property
    def values(self):
        """
        Gets the funnel data grouped by segments and calculated percentages.

        Returns:
            pd.DataFrame: The values of the funnel data with percentages calculated for each segment.
        """
        return self._funnel_data.groupby('segment') \
            .apply(self.calculate_percentage, include_groups=False)

    def plot(self,
             data: EventFrame,
             stages: List[Union[str, List[str]]],
             funnel_type: FunnelTypes = 'open',
             stages_names: Optional[List[str]] = None,
             inside_session: bool = False,
             segments: Optional[Iterable] = None,
             segments_names: Optional[Iterable] = None
             ) -> None:
        """
        Plots the funnel visualization using Plotly.

        Args:
            data (EventFrame): The event data to process.
            stages (List[Union[str, List[str]]]): The stages defined for the funnel.
            funnel_type (FunnelTypes, optional): The type of funnel to build ('open' or 'closed'). Defaults to 'open'.
            stages_names (Optional[List[str]], optional): The names for the stages. Defaults to None.
            inside_session (bool, optional): Indicator if the funnel should be calculated inside a session. Defaults to False.
            segments (Optional[Iterable], optional): The segments to analyze. Defaults to None.
            segments_names (Optional[Iterable], optional): The names for the segments. Defaults to None.

        Returns:
            None: Displays the funnel plot.
        """
        funnel_data = self.fit(data, stages, funnel_type, stages_names, inside_session, segments, segments_names)
        fig = go.Figure()
        for segment_name in funnel_data['segment'].unique():
            segment_data = funnel_data[funnel_data['segment'] == segment_name]
            fig.add_trace(go.Funnel(
                name=segment_name,
                orientation="h",
                y=segment_data['stage'],
                x=segment_data['users_count'],
                textinfo="value+percent initial+percent previous"
            ))

        fig.show()


def t():
    data = pd.read_csv("D:/diplom/simple_shop.csv")
    users = data['user_id'].unique()
    cities = [random.choice(['NY', 'SF', 'LA']) for _ in range(len(users))]
    users = pd.DataFrame({'user_id': users, 'city': cities})

    data = data.merge(users, on='user_id', how='inner')

    cols_schema = {
        'user_id': 'user_id',
        'event_name': 'event',
        'event_timestamp': 'timestamp'
    }

    ef = EventFrame(data, cols_schema=cols_schema, prepare=True)

    start_end_events_preprocessor = AddStartEndEventsPreprocessor()
    split_sessions_preprocessor = SplitSessionsPreprocessor(timeout=(15, 'm'))
    ef = start_end_events_preprocessor.apply(ef)
    ef = split_sessions_preprocessor.apply(ef)

    funnel = Funnel()
    funnel.fit(ef, funnel_type='closed', stages=[['product1', 'product2'], 'cart', 'payment_done'])


# t()
