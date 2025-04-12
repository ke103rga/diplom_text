import pandas as pd
from typing import Literal, Union, List, Optional, Dict
from sklearn import tree
import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod


ChoosingOptimalNClustersMethods = Literal['elbow', 'silhouette']


class ClusteringEstimator(ABC):
    default_init_params = None
    min_n_clusters = 2
    max_n_clusters = 100

    def _check_and_get_n_clusters(self, n_clusters: int = 5) -> int:
        if not isinstance(n_clusters, int) or n_clusters < self.min_n_clusters or n_clusters > self.max_n_clusters:
            raise ValueError('Invalid amount of clusters')
        return n_clusters

    @abstractmethod
    def fit(self, X: pd.DataFrame, sample_weight):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        pass

    @abstractmethod
    def fit_predict(self, X: pd.DataFrame, sample_weight):
        pass
    
    @staticmethod
    def plot_tree(tree_clf, data_cols):
        
        plt.figure()
        tree.plot_tree(tree_clf, filled=True, class_names=True, feature_names=data_cols)
        plt.title("Структура дерева решений кластеризации")
        plt.show()

    @staticmethod
    def plot_clusters_counts(estimator_labels):
        labels = pd.Series(estimator_labels)
        labels_counts = pd.merge(
            labels.value_counts().reset_index().rename(columns={'index': 'cluster'}),
            labels.value_counts(normalize=True).mul(100).round(2)\
                .reset_index().rename(columns={'index': 'cluster'}),
            on='cluster'
        ).sort_values(['cluster'])
        labels_counts['count_with_preportion'] = labels_counts['count'].astype(str) + ' (' + labels_counts['proportion'].astype(str) + ' %)'
        
        with sns.axes_style("darkgrid"):
            fig, axes = plt.subplots(1, 1, figsize=(8, 5))
            sns.barplot(data=labels_counts, x='cluster', y='count', ax=axes)

            for container in axes.containers:
                axes.bar_label(container,  labels=labels_counts['count_with_preportion'])

            axes.set_title("Распределение кластеров")
            axes.set_xlabel("Номер кластера")
            axes.set_ylabel("Количество объектов")

            plt.tight_layout()

    @staticmethod
    def plot_clusters_with_tsne(data, labels) -> None:
        """
        Рисует точечный график с использованием t-SNE, где цвет точки отражает принадлежность кластеру.
        """
        # Применяем t-SNE для снижения размерности до 2
        tsne = TSNE(n_components=2, random_state=42)
        data_tsne = tsne.fit_transform(data)

        # Определяем уникальные метки (кластеры)
        unique_labels = np.unique(labels)

        # Задаем цветовую палитру
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

        # Создаем график
        plt.figure(figsize=(10, 6))

        for i, label in enumerate(unique_labels):
            # Выбираем точки, принадлежащие текущему кластеру
            cluster_points = data_tsne[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        color=colors[i], label=f'Cluster {label}', alpha=0.5)

        plt.title('t-SNE Visualization of Clusters')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid()
        plt.show()
        

