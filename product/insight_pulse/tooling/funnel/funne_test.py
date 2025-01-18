import pandas as pd
import pytest
from tooling.funnel.funnel import Funnel, EventFrameColsSchema


# Функция для создания тестовых данных
def create_test_data():
    data = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4],
        'session_id': [1, 1, 1, 2, 2, 3, 3, 3, 4],
        'event_name': ['A', 'B', 'C', 'A', 'C', 'A', 'B', 'C', 'A'],
        'event_timestamp': pd.to_datetime([
            '2023-01-01 10:00', '2023-01-01 10:01', '2023-01-01 10:02',
            '2023-01-01 10:00', '2023-01-01 10:05',
            '2023-01-01 10:00', '2023-01-01 10:01', '2023-01-01 10:02',
            '2023-01-01 10:00'
        ])
    })
    cols_schema = EventFrameColsSchema({
        'user_id': 'user_id',
        'session_id': 'session_id',
        'event_name': 'event_name',
        'event_timestamp': 'event_timestamp'
    })
    return data, cols_schema


def test_fit_open_funnel():
    funnel = Funnel()
    data, cols_schema = create_test_data()

    # Правильный ответ (ожидаемая воронка)
    expected_funnel = pd.DataFrame({
        'stage': ['A', 'B', 'C'],
        'users_count': [3, 2, 3],
        'segment': ['all_users', 'all_users', 'all_users']
    })

    # Рассчет воронки
    funnel_data = funnel.fit(
        funnel_type='open',
        data=data,
        cols_schema=cols_schema,
        stages=['A', 'B', 'C'],
        stages_names=None,
        inside_session=False,
        segments=None,
        segments_names=None
    )
    print()
    print(data)
    print(funnel_data)
    print(expected_funnel)

    # Проверки
    pd.testing.assert_frame_equal(funnel_data, expected_funnel)


def test_fit_closed_funnel():
    funnel = Funnel()
    data, cols_schema = create_test_data()

    # Правильный ответ (ожидаемая воронка для закрытой воронки)
    expected_funnel = pd.DataFrame({
        'stage': ['A', 'B', 'C'],
        'users_count': [3, 2, 2],
        'segment': ['all_users', 'all_users', 'all_users']
    })

    # Рассчет воронки
    funnel_data = funnel.fit(
        funnel_type='closed',
        data=data,
        cols_schema=cols_schema,
        stages=['A', 'B', 'C'],
        stages_names=None,
        inside_session=False,
        segments=None,
        segments_names=None
    )

    # Проверки
    pd.testing.assert_frame_equal(funnel_data.reset_index(drop=True), expected_funnel)


if __name__ == '__main__':
    pytest.main()
