import pandas as pd
from typing import Optional
import numpy as np 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from eventframing.eventframe import EventFrame
from eventframing.cols_schema import EventFrameColsSchema
from eventframing.event_type import EventType


class ClusteringDataPreprocessor:

    def __init__(self):
        self.data = None
        self.cols_schema = None
        self.raw_cluster_matrix = None
        self.cluster_matrix = None
        self.cluster_matrix_cat_features_names = None
        self.data_preprocessor = None
        self.cat_features = None
        self.not_preprocess_cols = []

    # temporary interface without meta info
    def create_cluster_matrix(self, data: EventFrame,
                              add_session_stats: bool = True, add_path_stats: bool = True,
                              preprocess: bool = True) -> pd.DataFrame:
        return self._create_cluster_matrix(
            data, meta_info=None,
            add_session_stats=add_session_stats,
            add_path_stats=add_path_stats,
            preprocess=preprocess
        )
    
    def _create_cluster_matrix(self, data: EventFrame,
                            meta_info: Optional[pd.DataFrame] = None,
                            add_session_stats: bool = True, add_path_stats: bool = True,
                            preprocess: bool = True) -> pd.DataFrame:

        self._check_input_params(data, meta_info)

        self.data = data.data.copy()
        self.cols_schema = data.cols_schema

        self.not_preprocess_cols.append(self.cols_schema.user_id)

        self._prepare_raw_cluster_matrix(self.data, self.cols_schema, meta_info, add_session_stats, add_path_stats)
        if preprocess:
            self._create_preprocessor_pipeline()
            self._preprocess_raw_cluster_matrix()
            return self.cluster_matrix

        return self.raw_cluster_matrix

    def _check_input_params(
            self, 
            data: EventFrame,
            meta_info:pd.DataFrame):
        
        if not isinstance(data, EventFrame):
            raise ValueError('EventFrame only')
        
        cols_schema = data.cols_schema
        data = data.data
        user_id_col = cols_schema.user_id
        
        if meta_info is not None and set(data[user_id_col].unique()).difference(set(meta_info[user_id_col].unique())):
            print('There are some user_id in original data which are not in meta_info.\n It\'s necessary to fill missed values before clustering')
            # warnings.showwarning('There are some user_id in original data which are not in meta_info.\n It\'s necessary to fill missed values before clustering')
            # raise UserWarning('There are some user_id in original data which are not in meta_info.\n It\'s necessary to fill missed values before clustering')
        
    def _prepare_raw_cluster_matrix(
            self, data: pd.DataFrame, cols_schema: EventFrameColsSchema,
            meta_info: Optional[pd.DataFrame] = None,
            add_session_stats: bool = True, add_path_stats: bool = True) -> pd.DataFrame:
        user_id_col = cols_schema.user_id

        raw_cluster_matrix = self._count_users_actions_pivot_table(data, cols_schema)

        if add_session_stats:
            if cols_schema.session_id is None or data[data[cols_schema.event_type] == EventType.SESSION_START.value.name].empty:
                raise ValueError('EventFrameColsSchema must contain session_id and sessions info. Use SplitSessionsPreprocessor to add it')
            users_sessions_stats = self._get_sessions_stats(data, cols_schema)

            raw_cluster_matrix = pd.merge(
                raw_cluster_matrix,
                users_sessions_stats,
                how='inner',
                on=user_id_col
            )

        if add_path_stats:
            if data[data[cols_schema.event_type] == EventType.PATH_START.value.name].empty:
                raise ValueError('EventFrameColsSchema paths info. Use AddStartEndEvents class to add it')
            users_paths_stats = self._get_user_path_stats(data, cols_schema)

            raw_cluster_matrix = pd.merge(
                raw_cluster_matrix,
                users_paths_stats,
                how='inner',
                on=user_id_col
            )
        
        # TODO: add meta_info and encode categorical columns by mean encoder like in CatBoost
        if False and meta_info is not None:
            raw_cluster_matrix = pd.merge(
                raw_cluster_matrix,
                meta_info,
                how='left',
                on=user_id_col
            )

        self.raw_cluster_matrix = raw_cluster_matrix
        return raw_cluster_matrix

    def _count_users_actions_pivot_table(self, data: pd.DataFrame, cols_schema: EventFrameColsSchema) -> pd.DataFrame:
        user_id_col = cols_schema.user_id
        event_name_col = cols_schema.event_name
        event_id_col = cols_schema.event_id
        event_type_col = cols_schema.event_type

        return data[data[event_type_col] == EventType.RAW.value.name].pivot_table(
            columns=event_name_col,
            index=user_id_col,
            values=event_id_col,
            aggfunc='nunique'
        ).reset_index().fillna(0)
    
    def _get_sessions_stats(self, data: pd.DataFrame, cols_schema: EventFrameColsSchema) -> pd.DataFrame:
        user_id_col = cols_schema.user_id
        event_name_col = cols_schema.event_name
        event_type_col = cols_schema.event_type
        session_id_col = cols_schema.session_id
        dt_col = cols_schema.event_timestamp

        session_start_name = EventType.SESSION_START.value.name
        session_end_name = EventType.SESSION_END.value.name
        raw_event_type_name = EventType.RAW.value.name

        session_starts = data.loc[data[event_name_col] == session_start_name, 
                                  (user_id_col, session_id_col, dt_col)]\
                        .rename(columns={
                            dt_col: 'session_start_dt'
                        })
        session_ends = data.loc[data[event_name_col] == session_end_name, 
                                  (user_id_col, session_id_col, dt_col)]\
                        .rename(columns={
                            dt_col: 'session_end_dt'
                        })
        raw_session_stats = pd.merge(
            session_starts,
            session_ends,
            how='inner',
            on=[session_id_col, user_id_col]
        )
        raw_session_stats['duration'] = (raw_session_stats['session_end_dt'] - raw_session_stats['session_start_dt'])\
            .apply(lambda delta: delta.total_seconds()).div(60)
        
        raw_session_stats = raw_session_stats.sort_values([user_id_col, 'session_start_dt'])
        raw_session_stats['gap'] = (raw_session_stats['session_end_dt'] -\
            raw_session_stats.groupby(user_id_col)['session_start_dt'].shift())\
                .apply(lambda delta: delta.total_seconds()) / 60
        raw_session_stats['gap'] = raw_session_stats['gap'].fillna(0)
        
        raw_session_stats = pd.merge(
            raw_session_stats,
            data[data[event_type_col] == raw_event_type_name].groupby(session_id_col).agg(**{
                'amount_of_steps': (dt_col, 'count')
            }).reset_index(),
            how='inner',
            on=[session_id_col]
        )
        
        
        user_sessions_stats_aggs = {
            'amount_of_sessions': (session_id_col, 'count'),
        }
        for col in ['duration', 'gap', 'amount_of_steps']:
            for stat in ['min', 'max', 'mean', 'median']:
               user_sessions_stats_aggs[f'{stat}_{col}'] = (col, stat) 

        user_sessions_stats = raw_session_stats.groupby(user_id_col)\
            .agg(**user_sessions_stats_aggs).reset_index()


        return user_sessions_stats
    
    def _get_user_path_stats(self, data: pd.DataFrame, cols_schema: EventFrameColsSchema) -> pd.DataFrame:
        user_id_col = cols_schema.user_id
        event_name_col = cols_schema.event_name
        event_id_col = cols_schema.event_id
        event_type_col = cols_schema.event_type
        session_id_col = cols_schema.session_id
        dt_col = cols_schema.event_timestamp

        path_start_name = EventType.PATH_START.value.name
        path_end_name = EventType.PATH_END.value.name
        
        path_starts = data.loc[data[event_name_col] == path_start_name, 
                                  (user_id_col, dt_col)]\
                        .rename(columns={
                            dt_col: 'path_start_dt'
                        })
        path_ends = data.loc[data[event_name_col] == path_end_name, 
                                  (user_id_col, dt_col)]\
                        .rename(columns={
                            dt_col: 'path_end_dt'
                        })
        raw_path_stats = pd.merge(
            path_starts,
            path_ends,
            how='inner',
            on=[user_id_col]
        )

        # Rewrite to amount of days
        raw_path_stats['lifetime'] = (raw_path_stats['path_end_dt'] - raw_path_stats['path_start_dt'])\
            .apply(lambda delta: delta.total_seconds()).div(60)
        raw_path_stats = pd.merge(
            raw_path_stats,
            data.groupby(user_id_col).agg(**{
                'amount_of_active_days': (dt_col, lambda datetimes: datetimes.dt.date.nunique())
            }).reset_index(),
            how='inner',
            on=[user_id_col]
        )

        return raw_path_stats.drop(columns=['path_start_dt', 'path_end_dt'])
        
    
    def _create_preprocessor_pipeline(self) -> None: 

        raw_cluster_matrix = self.raw_cluster_matrix.drop(columns=self.not_preprocess_cols)

        num_features = raw_cluster_matrix.select_dtypes(include=np.number).columns.tolist()
        cat_features = raw_cluster_matrix.select_dtypes(include=["object"]).columns.tolist()

        num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        cat_transformer = Pipeline(steps=[("OHE", OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

        data_preprocessor = ColumnTransformer(
            transformers=[('num', num_transformer, num_features), ("cat", cat_transformer, cat_features)],
            remainder='passthrough'
        )
        self.cluster_matrix_cat_features_names = cat_features
        self.data_preprocessor = data_preprocessor
        self.cat_features = cat_features

    def _preprocess_raw_cluster_matrix(self,):
        cluster_matrix = self.data_preprocessor.fit_transform(
            self.raw_cluster_matrix.drop(columns=self.not_preprocess_cols)
        )

        # Get names of preprocessed data to create a dataframe for validating features
        new_num_names = self.data_preprocessor.transformers_[0][2].copy()
        new_names = np.array(new_num_names, dtype="object")
        if self.cat_features:
            new_cat_names = self.data_preprocessor.transformers_[1][1].named_steps['OHE']\
                .get_feature_names_out(self.cluster_matrix_cat_features_names).copy()
            new_names = np.concatenate(new_names, new_cat_names)

        cluster_matrix = pd.DataFrame(cluster_matrix, columns=new_names)
        for col_name in self.not_preprocess_cols:
            cluster_matrix[col_name] = self.raw_cluster_matrix[col_name]
        self.cluster_matrix = cluster_matrix

    def inverse_transform(self, data):
        # Извлечение только преобразованных данных
        not_preprocess_cols = []
        if self.not_preprocess_cols:
            not_preprocess_cols = set(self.not_preprocess_cols).intersection(set(data.columns))
            print(not_preprocess_cols)
            data_processed = data.drop(columns=not_preprocess_cols).copy()
        else:
            data_processed = data.copy()

        # reconstructed_data = pd.DataFrame()

        # Восстановление категориальных переменных
        cat_features_indices = self.data_preprocessor.transformers_[1][2]
        cat_features = self.cat_features
        if cat_features:
        
            # Инвертируем OneHotEncoding
            ohe = self.data_preprocessor.transformers_[1][1]
            cat_feature_names = ohe.get_feature_names_out(cat_features)

            # Получаем категориальные значения на основе обработки
            inverse_cat_matrix = ohe.inverse_transform(data_processed.loc[:, (cat_features)])
            
            # Создаем новый DataFrame
            reconstructed_data = pd.DataFrame(data_processed[:, :len(cat_features)],
                                           columns=cat_feature_names)
            for i, col in enumerate(cat_feature_names):
                reconstructed_data[col] = inverse_cat_matrix[:, i]

        # Восстанавливаем числовые переменные
        num_features = self.data_preprocessor.transformers_[0][2]
        num_scaler = self.data_preprocessor.transformers_[0][1]

        # Извлекаем отмасштабированные данные
        scaled_data = data_processed.loc[:, num_features]

        # Восстанавливаем данные с помощью inverse_transform
        reconstructed_data = pd.DataFrame(num_scaler.inverse_transform(scaled_data), columns=num_features).round(decimals=10)

        not_preprocess_cols = not_preprocess_cols.union(set(data_processed.columns) - set(reconstructed_data.columns))
        # Восстанавливаем не обрабатываемые колонки
        for col_name in not_preprocess_cols:
            reconstructed_data[col_name] = data[col_name]
            
        return reconstructed_data


