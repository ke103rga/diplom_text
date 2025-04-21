import pandas as pd
from typing import List, Optional,Dict
from sklearn.cluster import DBSCAN

from clustering.clustering_estimator import ClusteringEstimator, ChoosingOptimalNClustersMethods


class DBScanEstimator(ClusteringEstimator):
    default_init_params = {
        'eps': 0.5,
        'min_samples': 5
    }

    def __init__(self, eps: Optional[float] = None, min_samples: Optional[int] = None) -> None:
        self.eps = eps if eps is not None else self.default_init_params['eps']
        self.min_samples = min_samples if min_samples is not None else self.default_init_params['min_samples']
        self.estimator = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.data_cols = None

    def get_default_init_params(self):
        return self.default_init_params
    
    def _save_data_cols(self, X):
        if isinstance(X, pd.DataFrame):
            self.data_cols = X.columns

    def fit(self, X, sample_weight=None):
        self._save_data_cols(X)
        self.estimator.fit(X)
        return self.estimator
    
    def predict(self, X, sample_weight=None):
        self._save_data_cols(X)
        return self.estimator.predict(X)

    def fit_predict(self, X, sample_weight=None):
        self._save_data_cols(X)
        return self.estimator.fit_predict(X)

    def plot_clusters_counts(self):
        super().plot_clusters_counts(self.estimator.labels_)

    def plot_clusters_with_tsne(self, X):
        self.fit(X)
        super().plot_clusters_with_tsne(X, self.estimator.labels_)

    @property
    def labels_(self):
        return self.estimator.labels_