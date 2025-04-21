import pandas as pd
from typing import Literal, Union,  Optional, Dict, Tuple
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from clustering.clustering_estimator import ClusteringEstimator, ChoosingOptimalNClustersMethods



EstimatorInit = Literal['k-means++', 'random']


class KMeansEstimator(ClusteringEstimator):
    _default_init_params = {
        'n_clusters': 8,
        'init': 'k-means++',
        'n_init': 'auto',
        'max_iter': 1000   
    }

    def __init__(self, n_clusters: Optional[Union[str, int]] = None, init: Optional[EstimatorInit] = None,
                 n_init:  Optional[Union[str, int]] = None, max_iter: Optional[int] = None) -> None:
        
        self.n_clusters = super()._check_and_get_n_clusters(n_clusters)
        self.init = init if init is not None else self._default_init_params['init']
        self.n_init = n_init if n_init is not None else self._default_init_params['n_init']
        self.max_iter = max_iter if max_iter is not None else self._default_init_params['max_iter']

        self.estimator = KMeans(n_clusters=self.n_clusters, init=self.init,
                                 n_init=self.n_init, max_iter=self.max_iter)
        
        self.data_cols = None
        self.tree_clf = None
    
    def get_default_init_params(self):
        return self._default_init_params
    
    @staticmethod
    def choose_optimal_n_clusters(n_clusters_range: Tuple[int, int], X, sample_weight=None,
                                   init_params: Optional[Dict] = None, 
                                   method:ChoosingOptimalNClustersMethods = 'elbow'):
        # Check range
        n_clusters_list = list(range(n_clusters_range[0], n_clusters_range[1] + 1))
        
        if init_params is None:
            init_params = {
                'init': 'k-means++',
                'n_init': 'auto',
                'max_iter': 1000   
            }
        if 'n_clusters' in init_params.keys():
            del init_params['n_clusters']

        if method == 'elbow':
            inertia = []
            for n_clusters in n_clusters_list:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, **(init_params if init_params else {}))
                kmeans.fit(X=X, sample_weight=sample_weight)
                inertia.append(kmeans.inertia_)

            # Визуализация удачи метода локтя
            with sns.axes_style("darkgrid"):
                fig, axes = plt.subplots(figsize=(8, 5))
                sns.lineplot(x=n_clusters_list, y=inertia, marker='o', ax=axes)
                axes.set_title('Метод локтя')
                axes.set_xlabel('Количество кластеров')
                axes.set_ylabel('Сумма квадратов расстояний')
                axes.set_xticks(n_clusters_list)

        elif method == 'silhouette':
            scores = []
            for n_clusters in n_clusters_list:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, **(init_params if init_params else {}))
                kmeans.fit(X=X, sample_weight=sample_weight)
                labels = kmeans.predict(X)
                score = silhouette_score(X, labels)
                scores.append(score)

            # Визуализация метода силуэта
            with sns.axes_style("darkgrid"):
                fig, axes = plt.subplots(figsize=(8, 5))
                bar_plot = sns.barplot(x=n_clusters_list, y=scores, ax=axes)
                axes.set_title('Метод силуэта')
                axes.set_xlabel('Количество кластеров')
                axes.set_ylabel('Коэффициент силуэта')
                # axes.set_xticks(n_clusters_list)
                # axes.set_xticklabels(n_clusters_list)

            # Добавление подписей к столбцам
            for p in bar_plot.patches:
                bar_plot.annotate(f'{p.get_height():.2f}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha='center', va='bottom', fontsize=10)

            # plt.show()
        else:
            raise ValueError(f"Use 'elbow' or 'silhouette' as choosing method")
        
    def set_n_clusters(self, n_clusters):
        self.n_clusters = n_clusters
        self.estimator.set_params(**{'n_clusters': n_clusters})

    def _save_data_cols(self, X):
        if isinstance(X, pd.DataFrame):
            self.data_cols = X.columns

    def fit(self, X, sample_weight=None):
        self._save_data_cols(X)
        
        self.estimator.fit(X=X, sample_weight=sample_weight)
        return self.estimator
    
    def predict(self, X):
        self._save_data_cols(X)
        return self.estimator.predict(X)
    
    def fit_predict(self, X, sample_weight=None):
        self._save_data_cols(X)
        return self.estimator.fit_predict(X, sample_weight)
    
    def describe_by_centroids(self):
        centroids = pd.DataFrame(data=self.estimator.cluster_centers_, columns=self.data_cols)
        centroids['cluster'] = range(centroids.shape[0])

        columns_order = ['cluster'] + list(centroids.drop(columns=['cluster']).columns)
        return centroids[columns_order]

    def plot_clusters_counts(self):
        super().plot_clusters_counts(self.estimator.labels_)

    def plot_clusters_with_tsne(self, X):
        self.fit(X)
        super().plot_clusters_with_tsne(X, self.estimator.labels_)

    @property
    def labels_(self):
        return self.estimator.labels_


