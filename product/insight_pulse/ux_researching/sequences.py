import pandas as pd
from typing import Union, List, Optional, Tuple

from eventframing.eventframe import EventFrame
from eventframing.cols_schema import EventFrameColsSchema


class Sequences:
    def __init__(self, ef: EventFrame):
        self.ef = ef
        self.cols_schema = ef.cols_schema
        self.threshold = None
        self.sequences_df = None
        self.amount_of_sequences = 0
        self.ngram_range = None
        self.unique_weight_col_name = ''
        self.share_weight_col_name = ''
        self.intersect_separator = ' -> '
        self.following_separator = ' => '

    def _get_data_and_schema(self, data: Optional[Union[EventFrame, pd.DataFrame]] = None,
                             cols_schema: Optional[EventFrameColsSchema] = None) -> Tuple[
        pd.DataFrame, EventFrameColsSchema]:
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

    def select_freq_sets(self, data: Optional[EventFrame] = None, ngram_range: Tuple[int, int] = (2, 3),
                         support_threshold: float = 0.05, inside_session: bool = True):
        data, cols_schema = self._get_data_and_schema(data=data)
        user_col = cols_schema.user_id
        event_col = cols_schema.event_name
        session_col = cols_schema.session_id
        dt_col = cols_schema.event_timestamp

        if inside_session and cols_schema.session_id is None:
            raise ValueError('EventFrame does not have session_id column.')
        weight_col = session_col if inside_session else user_col
        unique_weight_col_name = f'{weight_col}_unique'
        share_weight_col_name = f'{weight_col}_share'

        # Sort data and add 'number of step in path or session' column
        data = data.sort_values(by=[user_col, dt_col])
        data['step'] = data.groupby(weight_col)[dt_col].cumcount() + 1

        # Pivot table to create a matrix of events on every step in path or session
        data_pivot = data.pivot_table(index=weight_col, columns='step', values=event_col, aggfunc=lambda x: x)

        # Create a list of all possible ngrams
        # In addition to parametr's range use (ngram_range[0] - 1) to compute count of premise for associative rules
        ngram_list = list(range(ngram_range[0] - 1, ngram_range[1] + 1)) + [1]
        ngram_list = [x for x in ngram_list if x > 0]  # Remove values less than 1
        ngram_list = list(set(ngram_list))  # Remove duplicates

        # Generate sequences as a list of combination of events for each user or session
        sequences = []
        for weight_col_idx, row in data_pivot.iterrows():
            events = row.dropna().astype(str).tolist()
            for n in ngram_list:
                for i in range(len(events) - n + 1):
                    sequences.append([weight_col_idx, self.intersect_separator.join(events[i:i + n]), n])

        # Convert sequences to DataFrame
        sequences_df = pd.DataFrame(sequences, columns=['weight_col_id', 'sequence', 'sequence_len'])
        # Count the frequency of each sequence and the number of unique users or sessions they are associated with
        sequences_df = sequences_df.groupby('sequence').agg(**{
            'count': ('weight_col_id', 'count'),
            unique_weight_col_name: ('weight_col_id', 'nunique'),
            'sequence_len': ('sequence_len', 'first')
        }).reset_index()
        # Compute support and user/session share
        amount_of_sequences = sequences_df[sequences_df['sequence_len'] >= ngram_range[0]]['count'].sum()
        sequences_df['support'] = sequences_df['count'] / amount_of_sequences
        sequences_df[share_weight_col_name] = sequences_df[unique_weight_col_name] / data_pivot.shape[0]

        # Save sequences_df to use in searching associative rules
        self.sequences_df = sequences_df
        self.amount_of_sequences = amount_of_sequences
        self.threshold = support_threshold
        self.unique_weight_col_name = unique_weight_col_name
        self.share_weight_col_name = share_weight_col_name
        self.ngram_range = ngram_range
        return sequences_df[
            (sequences_df['sequence_len'] >= ngram_range[0]) &
            (sequences_df['support'] >= support_threshold)
            ].drop(columns=['sequence_len'])

    def associative_rules(self, target_events: Optional[Union[str, List[str]]] = None):
        if self.sequences_df is None:
            raise ValueError('It is necessary to use select_freq_sets method first.')

        # Filter the frequent itemsets based on the minimum support threshold
        frequent_sequences = self.sequences_df[
            (self.sequences_df['support'] > self.threshold) &
            (self.sequences_df['sequence_len'] > self.ngram_range[0])
        ].copy()

        # Split sequences into premise and conclusion
        frequent_sequences['split_index'] = frequent_sequences['sequence'].str.rfind(self.intersect_separator)
        frequent_sequences['premise'] = frequent_sequences.apply(lambda row: row['sequence'][:row['split_index']],
                                                                 axis=1)
        frequent_sequences['conclusion'] = frequent_sequences.apply(
            lambda row: row['sequence'][row['split_index'] + len(self.intersect_separator):],
            axis=1
        )

        # Filter the frequent itemsets based on the target events
        if isinstance(target_events, str):
            target_events = [target_events]
        elif target_events is None:
            target_events = []

        if len(target_events) > 0:
            frequent_sequences = frequent_sequences[frequent_sequences['conclusion'].isin(target_events)]

        # Merge the count of premise and conclusion for each frequent itemset
        frequent_sequences = pd.merge(
            frequent_sequences,
            self.sequences_df.loc[:, ('sequence', 'count')] \
                .rename(columns={'sequence': 'premise', 'count': 'count_premise'}),
            how='inner',
            on='premise'
        )
        frequent_sequences = pd.merge(
            frequent_sequences,
            self.sequences_df.loc[:, ('sequence', 'count')] \
                .rename(columns={'sequence': 'conclusion', 'count': 'count_conclusion'}),
            how='inner',
            on='conclusion'
        )
        # Compute the confidence and lift of each frequent itemset
        frequent_sequences['confidence'] = frequent_sequences['count'] / frequent_sequences['count_premise']
        frequent_sequences['lift'] = frequent_sequences['count'] \
            .div(frequent_sequences['count_conclusion'] * frequent_sequences['count_premise']) \
            .mul(self.amount_of_sequences)

        # Format result DataFrame
        frequent_sequences['rule'] = frequent_sequences['premise'].astype(str) \
                                     + self.following_separator \
                                     + frequent_sequences['conclusion']

        result_cols = ['rule', 'count', 'support', 'confidence', 'lift', self.unique_weight_col_name,
                       self.share_weight_col_name]
        return frequent_sequences.loc[:, result_cols]