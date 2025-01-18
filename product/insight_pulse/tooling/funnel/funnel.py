from typing import Literal, Union, List, Optional, Iterable, get_args
import pandas as pd
from plotly import graph_objects as go
import plotly.express as px


from eventframing.cols_schema import EventFrameColsSchema
from eventframing.eventframe import EventFrame


FunnelTypes = Literal["open", "closed"]


class Funnel:
    _funnel_data: pd.DataFrame

    def __init__(self):
        pass

    def _check_fit_params(
            self,
            funnel_type: FunnelTypes,
            data: Union[pd.DataFrame, 'EventFrame'],
            cols_schema: Optional[EventFrameColsSchema],
            stages: List[Union[str, List[str]]],
            stages_names: Optional[List[str]],
            inside_session: bool,
            segments: Optional[Iterable],
            segments_names: Optional[Iterable]
    ):
        if funnel_type not in get_args(FunnelTypes):
            raise ValueError(f'funnel_type should be one of {get_args(FunnelTypes)}')

        if not (isinstance(data, EventFrame) or
                (isinstance(data, pd.DataFrame) and isinstance(cols_schema, EventFrameColsSchema))):
            raise ValueError('EventFrame or DataFrame with EventFrameColsSchema')

        if not len(set(stages)) == len(stages):
            raise ValueError(f'All stages should be distinct')

        if stages_names:
            if not len(set(stages_names)) == len(stages_names):
                raise ValueError(f'All stages names should be distinct')

            if not len(stages) == len(stages_names):
                raise ValueError(f'Amount of Stages differ from amount of their names')

        if segments:
            if self._segments_repeated_indexes(segments):
                raise ValueError(f'Segments should not contain repeating elements')

            if segments_names and not len(segments) == len(segments_names):
                raise ValueError(f'Amount of segments differ from amount of their names')

    def _segments_repeated_indexes(self, segments):
        segment_idx = set()

        for segment in segments:
            segment_set = set(segment)
            if len(segment_idx.intersection(segment_set)) > 0:
                return False
            segment_idx = segment_idx.union(segment_set)
        return True

    def _collapse_stages(self, data: pd.DataFrame, cols_schema: EventFrameColsSchema,
                         stages: List[Union[str, List[str]]]) -> pd.DataFrame:

        multiple_stages = [stage for stage in stages if isinstance(stage, list)]
        if len(multiple_stages) == 0:
            return data

        data_copy = data.copy()
        event_col = cols_schema.event_name
        multiple_stages_names = [mult_stage[0] for mult_stage in multiple_stages]

        collapse_dict = dict()
        for multiple_stage, multiple_stage_name in zip(multiple_stages, multiple_stages_names):
            for stage in multiple_stage:
                collapse_dict[stage] = multiple_stage_name

        data_copy[event_col].replace(collapse_dict, inpace=True)

        return data_copy

    def fit(
            self,
            funnel_type: FunnelTypes,
            data: Union[pd.DataFrame, 'EventFrame'],
            cols_schema: Optional[EventFrameColsSchema],
            stages: List[Union[str, List[str]]],
            stages_names: Optional[List[str]] = None,
            inside_session: bool = True,
            segments: Optional[Iterable] = None,
            segments_names: Optional[Iterable] = None
    ) -> pd.DataFrame:
        self._check_fit_params(funnel_type, data, cols_schema, stages,
                               stages_names, inside_session, segments, segments_names)
        self._collapse_stages(data, cols_schema, stages)

        if isinstance(data, EventFrame):
            data = data.data.copy()
            cols_schema = data.cols_schema

        user_id_col = cols_schema.user_id
        event_col = cols_schema.event_name

        funnel_data = pd.DataFrame(columns=['stage', 'users_count', 'segment'])

        if segments is None:
            segments, segments_names = [data.index], ['all_users']

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
            segment_funnel_data.sort_values(by='stage_rank', ascending=True, inplace=True)
            segment_funnel_data.drop(columns=['stage_rank'], inplace=True)

            segment_funnel_data['segment'] = segment_name

            funnel_data = pd.concat([funnel_data, segment_funnel_data])

        if stages_names is not None:
            funnel_data['stage'].replace(
                {stage: stage_name for stage, stage_name in zip(stages, stages_names)},
                inplace=True
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

        funnel_data.rename(columns={
            event_col: 'stage',
            user_id_col: 'users_count'
        }, inplace=True)

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
                print(stage)
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
                ).fillna(pd.to_datetime('2090-01-01', yearfirst=True)).reset_index()
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
        # Расчет процента от предыдущей строки
        data['percent_of_previous'] = data['users_count'] / data['users_count'].shift(1) * 100
        data['percent_of_previous'] = data['percent_of_previous'].fillna(100)

        # Расчет процента от первого значения в группе
        data['percent_of_initial'] = data['users_count'] / data['users_count'].iloc[0] * 100

        return data

    @property
    def values(self):
        return self._funnel_data.groupby('segment')\
            .apply(self.calculate_percentage, include_groups=False)

    def plot(self):
        if self._funnel_data is None:
            raise RuntimeError('Previously must fit funnel')
        funnel_data = self._funnel_data
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




def create_test_data():
    data = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        'dt': [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        'session_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
        'event_name': ['A', 'B', 'C', 'A', 'C', 'D', 'A', 'B', 'D',  'C']
    })
    cols_schema = EventFrameColsSchema({
        'user_id': 'user_id',
        'event_timestamp': 'dt',
        'event_name': 'event_name',
        'session_id': 'session_id'
    })
    return EventFrame(data, cols_schema)


def test():
    funnel = Funnel()
    eventframe = create_test_data()
    data, cols_schema = eventframe.to_dataframe(), eventframe.cols_schema

    print(data)

    funnel = Funnel()
    funnel.fit(
        funnel_type='open',
        data=data,
        cols_schema=cols_schema,
        stages=['A', 'B', 'C'],
        inside_session=False,
    )
    print(funnel.values)
    funnel.plot()


# test()
