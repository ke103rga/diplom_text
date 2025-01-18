import pandas as pd

from data_preprocessing.preprocessors_lib.add_start_end_events import AddStartEndEventsPreprocessor
from data_preprocessing.preprocessors_lib.split_sessions import SplitSessionsPreprocessor
from eventframing.eventframe import EventFrame


def create_frame():
    df2 = pd.DataFrame(
        [
            ['user_1', 'A', '2023-01-01 00:00:00'],
            ['user_1', 'B', '2023-01-01 00:00:05'],
            ['user_2', 'B', '2023-01-01 00:00:02'],
            ['user_2', 'A', '2023-01-01 00:00:03'],
            ['user_2', 'A', '2023-01-01 00:00:07']
        ],
        columns=['client_id', 'action', 'datetime']
    )

    raw_data_schema = {
        'user_id': 'client_id',
        'event_name': 'action',
        'event_timestamp': 'datetime'
    }

    frame = EventFrame(df2, cols_schema=raw_data_schema)
    return frame


def test_filter(frame):
    conditions = [
        "client_id == 'user_1'",
        "action.isin(['A', 'B'])"
    ]
    print(f"Frame after filtering by conditions: {conditions}")
    print(frame.filter(conditions).to_dataframe(), end='\n\n')


def test_AddStartEndEventsPreprocessor(frame):
    pr = AddStartEndEventsPreprocessor()
    print('Apply preprocessor to data and schema:')
    print(pr.apply(frame.data, frame.cols_schema).to_dataframe().drop(columns=['action', 'event_id']), end='\n\n')

    print('Apply preprocessor directly to frame:')
    print(pr.apply(frame).to_dataframe().drop(columns=['action', 'event_id']), end='\n\n')


def test_SplitSessionsPreprocessor(frame):
    pr = SplitSessionsPreprocessor(timeout=(1, 's'))

    print('Apply preprocessor directly to frame:')
    print(pr.apply(frame).to_dataframe().drop(columns=['action', 'client_id', 'event_type_index']), end='\n\n')


def test_both_preprocessors(frame):
    pr1 = AddStartEndEventsPreprocessor()
    pr2 = SplitSessionsPreprocessor(timeout=(1, 's'))

    print('Apply both preprocessors:')
    frame = pr1.apply(frame)
    frame = pr2.apply(frame)
    print(frame.to_dataframe().drop(columns=['action', 'client_id', 'event_type_index']), end='\n\n')


def test():
    frame = create_frame()
    print("Frame before manipulations:")
    print(frame)

    # test_filter(frame)

    # test_AddStartEndEventsPreprocessor(frame)

    # test_SplitSessionsPreprocessor(frame)

    test_both_preprocessors(frame)


test()
