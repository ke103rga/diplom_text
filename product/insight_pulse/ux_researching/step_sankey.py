import pandas as pd
from typing import Union, List, Optional, Tuple
import plotly.graph_objects as go
from pandas import DataFrame
import seaborn as sns

from eventframing.eventframe import EventFrame
from eventframing.cols_schema import EventFrameColsSchema


class StepSankey:
    # TODO: rename param weight_col into `inside_session`: bool and drop path start events if it's true
    _path_end_event_name = 'ENDED'

    def __init__(self, ef: EventFrame):
        self.ef = ef
        self.cols_schema = ef.cols_schema
        self.weight_col = ''
        self.total_weight_col = ''
        self.step_weight_col = ''
        self.link_weight_col = ''
        self.nodes = None
        self.links = None

    def fit(self, data: Optional[EventFrame] = None, max_steps: int = 10, threshold: float = 0.05,
            weight_col: str = '', events_to_keep: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data, _ = self._get_data_and_schema(data=data)
        prepared_data = self._prepare_data(data, max_steps, weight_col)
        nodes, rare_events = self._get_nodes(prepared_data, threshold, events_to_keep)
        links = self._get_links(prepared_data, nodes, rare_events)
        self.nodes = nodes
        self.links = links
        return nodes, links

    def plot(self, data: Optional[EventFrame] = None, max_steps: int = 10, threshold: float = 0.05,
             weight_col: str = '', events_to_keep: Optional[List[str]] = None, title: str = 'StepSankey'):
        data, cols_schema = self._get_data_and_schema(data=data)
        event_col = self.cols_schema.event_name
        nodes, links = self.fit(data, max_steps, threshold, weight_col, events_to_keep)

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes.step.astype(str) + nodes[event_col].astype(str),
                customdata=nodes.desc,
                hovertemplate='%{customdata}',
                color=nodes.color
            ),
            link=dict(
                source=links.source_index,
                target=links.target_index,
                value=links[self.link_weight_col],
                hovertemplate='Step from %{source.label}<br />' +
                              'to %{target.label}<br />made %{value} users',
            ))])

        fig.update_layout(title_text=title, font_size=10)
        fig.show()

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

    def _get_next_event(self, data: pd.DataFrame, weight_col: str, event_col: str) -> pd.DataFrame:
        grouped = data.groupby(weight_col)
        data['next_event'] = grouped[event_col].shift(-1)
        return data

    def _prepare_data(self, data: pd.DataFrame, max_steps: int, weight_col: str):
        data = data.copy()

        user_col = self.cols_schema.user_id
        event_col = self.cols_schema.event_name
        session_col = self.cols_schema.session_id
        dt_col = self.cols_schema.event_timestamp

        if weight_col != '':
            if weight_col not in data.columns:
                raise ValueError(f'Column {weight_col} is not in the EventFrame.')
        else:
            weight_col = session_col if session_col is not None else user_col
        self.weight_col = weight_col
        self.total_weight_col = f'total_{weight_col}'
        self.step_weight_col = f'step_{weight_col}'
        self.link_weight_col = f'link_{weight_col}'

        # Add column with number of step in session or the whole path
        data = data.sort_values(by=[user_col, dt_col])
        data['step'] = data.groupby(weight_col)[dt_col].cumcount() + 1
        data = data[data['step'] <= max_steps]

        # Add path terminating event if user has less steps than max_steps
        data = data.pivot_table(index=weight_col, columns='step', values=event_col, aggfunc=lambda x: x) \
            .fillna(self._path_end_event_name).reset_index()

        # Unpivot table into original format but with terminating event
        data = data.melt(id_vars=weight_col, var_name='step', value_name=event_col)
        # Add info about next event in path or session
        data = self._get_next_event(data, weight_col, event_col)
        return data

    def _get_nodes(self, prepared_data: pd.DataFrame, threshold: Union[float, int] = 0,
                   events_to_keep: List[str] = None) -> tuple[DataFrame, DataFrame]:
        event_col = self.cols_schema.event_name
        if events_to_keep is None:
            events_to_keep = [self._path_end_event_name]
        else:
            events_to_keep.append(self._path_end_event_name)

        nodes = prepared_data.groupby(by=['step', event_col]).agg(**{
            self.step_weight_col: (self.weight_col, 'nunique')
        }).reset_index()

        total_weight_col_value = prepared_data.loc[prepared_data['step'] == 1, self.weight_col].nunique()
        nodes[self.total_weight_col] = total_weight_col_value
        nodes['pers_of_total'] = nodes[self.step_weight_col].divide(nodes[self.total_weight_col])

        threshold_metric = self.step_weight_col if isinstance(threshold, int) else 'pers_of_total'
        events_to_keep.extend(self._path_end_event_name)
        rare_events = nodes[(nodes[threshold_metric] < threshold) & (~nodes[event_col].isin(events_to_keep))]

        if not rare_events.empty:
            rare_events_idx = rare_events.index
            rare_events_replacers = rare_events.groupby('step').agg(**{
                'pers_of_total': ('pers_of_total', 'sum'),
                self.step_weight_col: (self.step_weight_col, 'sum'),
                event_col: (event_col, lambda col: f'thresholded_{col.count()}')
            }).reset_index()
            rare_events_replacers[self.total_weight_col] = total_weight_col_value

            nodes = nodes.drop(index=rare_events_idx)
            nodes = pd.concat([
                nodes,
                rare_events_replacers,
            ], axis=0)

            rare_events = pd.merge(
                rare_events.loc[:, ('step', event_col)],
                rare_events_replacers.loc[:, ('step', event_col)].rename(columns={event_col: 'new_event_name'}),
                on='step'
            )

        all_events = nodes[event_col].unique()
        palette = self._prepare_palette(all_events, event_col)

        nodes = pd.merge(
            nodes,
            palette,
            on=event_col,
            how='inner'
        )

        nodes = nodes.sort_values(by=['step', self.step_weight_col], ascending=[True, True])
        nodes['index'] = list(range(nodes.shape[0]))
        nodes['desc'] = nodes.apply(lambda node: self._get_node_description(node, event_col), axis=1)
        return nodes, rare_events

    def _get_node_description(self, node, event_col):
        desc = node[event_col] + ' ' + str(node[self.step_weight_col]) + \
               ' (' + str(round(node['pers_of_total'] * 100, 1)) + '% of total)'
        return desc

    def _replace_links_rare_events(self, links: pd.DataFrame, rare_events: pd.DataFrame) -> pd.DataFrame:
        if rare_events.empty:
            return links

        event_col = self.cols_schema.event_name
        replaced_links = pd.merge(
            links,
            rare_events,
            on=['step', event_col],
            how='left'
        )

        replaced_links = pd.merge(
            replaced_links.assign(next_step=lambda x: x['step'] + 1),
            rare_events.rename(columns={event_col: 'next_event', 'step': 'next_step',
                                        'new_event_name': 'new_next_event_name'}),
            on=['next_step', 'next_event'],
            how='left'
        )

        replaced_links[event_col] = replaced_links['new_event_name'].fillna(replaced_links[event_col])
        replaced_links['next_event'] = replaced_links['new_next_event_name'].fillna(replaced_links['next_event'])
        # replaced_links = replaced_links.drop(columns=['new_event_name', 'new_next_event_name'])
        return replaced_links

    def _get_links(self, prepared_data: pd.DataFrame, nodes: pd.DataFrame, rare_events: pd.DataFrame) -> pd.DataFrame:
        event_col = self.cols_schema._event_name

        links = prepared_data.groupby(by=['step', event_col, 'next_event']).agg(**{
            self.link_weight_col: (self.weight_col, 'nunique')
        }).reset_index()

        links = self._replace_links_rare_events(links, rare_events)

        links = pd.merge(
            links,
            nodes,
            on=['step', event_col],
            how='inner'
        )
        links['pers_of_step'] = links[self.link_weight_col].divide(links[self.step_weight_col])
        links = links.rename(columns={'index': 'source_index'})

        links = pd.merge(
            links.assign(next_step=lambda x: x['step'] + 1),
            nodes\
                .rename(columns={event_col: 'next_event', 'step': 'next_step'})\
                .loc[:, ('next_step', 'next_event', 'index')],
            on=['next_step', 'next_event'],
            how='left'
        )
        links = links.rename(columns={'index': 'target_index'})
        return links

    @staticmethod
    def _prepare_palette(all_events: list, event_col: str) -> pd.DataFrame:
        palette_hex = [
            "50BE97",  # Нежный зеленый
            "E4655C",  # Красный
            "FCC865",  # Яркий желтый
            "BFD6DE",  # Светло-голубой
            "3E5066",  # Темный синий
            "353A3E",  # Темный серый
            "E6E6E6",  # Светло-серый
            "6D4C41",  # Коричневый
            "FFD54F",  # Лимонно-желтый
            "4DB6AC"  # Бирюзовый
        ]
        # convert HEX to RGB
        palette = []
        for color in palette_hex:
            rgb_color = tuple(int(color[i: i + 2], 16) for i in (0, 2, 4))
            palette.append(rgb_color)

        # extend color palette if number of events more than default colors list
        complementary_palette = sns.color_palette("deep", len(all_events) - len(palette))
        if len(complementary_palette) > 0:
            colors = complementary_palette.as_hex()
            for c in colors:
                col = c[1:]
                palette.append(tuple(int(col[i: i + 2], 16) for i in (0, 2, 4)))

        palette = pd.DataFrame({event_col: all_events, 'color': palette})
        palette['color'] = 'rgb' + palette['color'].astype(str).str.replace(' ', '')
        return palette
