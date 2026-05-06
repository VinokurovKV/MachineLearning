import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''

    _, inverse_labels, cluster_sizes = np.unique(
        labels,
        return_inverse=True,
        return_counts=True
    )
    n_objects = x.shape[0]
    n_clusters = cluster_sizes.shape[0]

    if n_clusters == 1:
        sil_score = 0.0
    else:
        distances = sklearn.metrics.pairwise_distances(x)

        order = np.argsort(inverse_labels)
        sorted_labels = inverse_labels[order]
        cluster_starts = np.r_[
            0,
            np.flatnonzero(np.diff(sorted_labels)) + 1
        ]

        cluster_distances_sum = np.add.reduceat(
            distances[:, order],
            cluster_starts,
            axis=1
        )
        mean_distances = cluster_distances_sum / cluster_sizes

        same_cluster_distances = cluster_distances_sum[
            np.arange(n_objects),
            inverse_labels
        ]
        same_cluster_sizes = cluster_sizes[inverse_labels] - 1

        compactness = np.divide(
            same_cluster_distances,
            same_cluster_sizes,
            out=np.zeros(n_objects, dtype=float),
            where=same_cluster_sizes != 0
        )

        mean_distances[np.arange(n_objects), inverse_labels] = np.inf
        separation = np.min(mean_distances, axis=1)

        denominator = np.maximum(compactness, separation)
        silhouettes = np.divide(
        separation - compactness,
        denominator,
        out=np.zeros(n_objects, dtype=float),
        where=denominator != 0
    )
    silhouettes[cluster_sizes[inverse_labels] == 1] = 0.0

    sil_score = float(np.mean(silhouettes))

    return sil_score


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''

    same_true = true_labels[:, None] == true_labels[None, :]
    same_predicted = predicted_labels[:, None] == predicted_labels[None, :]
    same_both = same_true & same_predicted

    precision = np.mean(
        same_both.sum(axis=1) / same_predicted.sum(axis=1)
    )
    recall = np.mean(
        same_both.sum(axis=1) / same_true.sum(axis=1)
    )

    score = 2 * precision * recall / (precision + recall)

    return score