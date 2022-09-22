import time
from sympy.utilities.iterables import multiset_permutations
from sklearn.metrics import homogeneity_score
from sklearn_extra.cluster import KMedoids
import concepts_xai.evaluation.metrics.purity as purity
import sklearn
import scipy
import tensorflow as tf
import numpy as np
from collections import defaultdict

def global_feat_importance_from_masks(
    c_train,
    masks
):
    masks = np.array(masks)
    acc = np.zeros(masks.shape[1])
    for i in range(c_train.shape[0]):
        for concept_idx, val in enumerate(c_train[i, :]):
            if val == 1:
                acc += masks[concept_idx, :]
    # And don't forget to normalize scores
    return acc / (c_train.shape[0] * len(masks))

def feature_importance_diff(
    importance_masks, # List[np.arrays of normalized scores for each feature] for each concept
    c_train,
    ground_truth_concept_masks,
):
    
    ground_truth_global = global_feat_importance_from_masks(
        c_train=c_train,
        masks=ground_truth_concept_masks,
    )
    importance_masks = np.array(importance_masks)
    if len(importance_masks.shape) > 1:
        importance_masks = np.mean(np.array(importance_masks), axis=0)
    return sklearn.metrics.mean_squared_error(
        ground_truth_global,
        importance_masks,
    )

def feature_selection(
    importance_masks, # List[np.arrays of normalized scores for each feature] for each concept
    c_train,
    ground_truth_concept_masks,
    threshold_ratio=list(np.arange(0.0, 1.0, 0.025)), #0.2,
):
    
    ground_truth_global = global_feat_importance_from_masks(
        c_train=c_train,
        masks=ground_truth_concept_masks,
    )
    ground_truth_global = (ground_truth_global > 0).astype(np.int32)
    importance_masks = np.array(importance_masks)
    if len(importance_masks.shape) > 1:
        importance_masks = np.mean(np.array(importance_masks), axis=0)
    if isinstance(threshold_ratio, (list, tuple, np.ndarray)):
        best = 0.0
        for ratio in threshold_ratio:
            thresh = np.max(importance_masks) * ratio
            best = max(
                best,
                sklearn.metrics.roc_auc_score(
                    ground_truth_global,
                    (importance_masks > thresh).astype(np.int32),
                )
            )
        return best
    thresh = np.max(importance_masks) * threshold_ratio
    return sklearn.metrics.roc_auc_score(
        ground_truth_global,
        (importance_masks > thresh).astype(np.int32),
    )


def brute_force_concept_mask_aucs(
    concept_importance_masks, # List[np.arrays of normalized scores for each feature] for each concept
    ground_truth_concept_masks,
    reduction=np.mean,
    thresh=None,
    alignment=None,
):
    result = {}
    concept_importance_masks = np.array(concept_importance_masks)
    ground_truth_concept_masks = np.array(ground_truth_concept_masks)
    if alignment is not None:
        perms = [alignment]
    else:
        perms = multiset_permutations(
            list(range(concept_importance_masks.shape[0])),
            size=min(
                ground_truth_concept_masks.shape[0],
                concept_importance_masks.shape[0],
            ),
        )
    for permutation in perms:
        current_masks = concept_importance_masks[permutation, :]
        current_aucs = []
        for i in range(min(
            ground_truth_concept_masks.shape[0],
            concept_importance_masks.shape[0],
        )):
            mask = current_masks[i, :]
            if thresh is not None:
                mask = (mask >= thresh).astype(np.int32)
            ground_truth = ground_truth_concept_masks[i, :]
            current_aucs.append(sklearn.metrics.roc_auc_score(
                ground_truth,
                mask,
            ))
        red_score = reduction(current_aucs)
        if result.get('best_reduced_auc', 0.0) < red_score:
            result['best_reduced_auc'] = red_score
            result['best_individual_auc'] = current_aucs
            result['best_permutation'] = permutation
    return result

def brute_force_concept_aucs(
    concept_scores,
    c_test,
    reduction=np.mean,
    thresh=None,
    alignment=None,
):
    result = {}
    if alignment is not None:
        perms = [alignment]
    else:
        perms = multiset_permutations(
            list(range(c_test.shape[-1])),
            size=min(
                c_test.shape[-1],
                concept_scores.shape[-1],
            ),
        )
    for permutation in perms:
        scores = concept_scores[:, permutation]
        current_aucs = []
        for i in range(min(
            c_test.shape[-1],
            concept_scores.shape[-1],
        )):
            # We take the max as the concept may be flipped
            current_score = max(
                sklearn.metrics.roc_auc_score(
                    c_test[:, i],
                    scores[:, i],
                ),
                sklearn.metrics.roc_auc_score(
                    c_test[:, i],
                    1 - scores[:, i],
                ),
            )
            current_aucs.append(current_score)
        red_score = reduction(current_aucs)
        if result.get('best_reduced_auc', 0.0) < red_score:
            result['best_reduced_auc'] = red_score
            result['best_individual_auc'] = current_aucs
            result['best_permutation'] = permutation
    return result

def find_best_independent_alignment(scores, c_train):
#     purity_mat = purity.concept_purity_matrix(
#         c_soft=scores,
#         c_true=c_train,
#     )
#     align = purity.find_max_alignment(purity_mat.T)
#     return align, purity_mat[list(range(0, scores.shape[-1])), align]
#     return np.argmax(purity_mat, axis=0), np.max(purity_mat, axis=0)
    n_concepts = scores.shape[-1]
    purity_mat = np.abs(np.corrcoef(np.hstack([scores, c_train]).T)[:n_concepts, n_concepts:])
    return np.argmax(purity_mat, axis=1), np.max(purity_mat, axis=1)

def correlation_alignment(scores, c_test):
    n_concepts = scores.shape[-1]
    return np.abs(np.corrcoef(np.hstack([scores, c_test]).T)[:n_concepts, n_concepts:])

def embedding_homogeneity(
    c_vec,
    c_test,
    y_test,
    step,
    force_alignment=True,
    alignment=None,
):
    """
    Computes the alignment between learnt concepts and labels.

    :param c_vec: predicted concept representations (can be concept embeddings)
    :param c_test: concept ground truth labels
    :param y_test: task ground truth labels
    :param step: integration step
    :return: concept alignment AUC, task alignment AUC
    """
    
    # First lets compute an alignment between concept
    # scores and ground truth concepts
    if force_alignment:
        if alignment is None:
            purity_mat = purity.concept_purity_matrix(
                c_soft=c_vec,
                c_true=c_test,
            )
            alignment = purity.find_max_alignment(purity_mat)
        # And use the new vector with its corresponding alignment
        c_vec = c_vec[:, alignment]
    
    # compute the maximum value for the AUC
    n_clusters = np.linspace(2, c_vec.shape[0], step).astype(int)
    max_auc = np.trapz(np.ones(step))

    # for each concept:
    #   1. find clusters
    #   2. compare cluster assignments with ground truth concept/task labels
    concept_auc, task_auc = [], []
    for concept_id in range(c_test.shape[1]):
        concept_homogeneity, task_homogeneity = [], []
        for nc in n_clusters:
            kmedoids = KMedoids(n_clusters=nc, random_state=0)
            if c_vec.shape[1] != c_test.shape[1]:
                c_cluster_labels = kmedoids.fit_predict(
                    np.hstack([
                        c_vec[:, concept_id][:, np.newaxis],
                        c_vec[:, c_test.shape[1]:]
                    ])
                )
            elif c_vec.shape[1] == c_test.shape[1] and len(c_vec.shape) == 2:
                c_cluster_labels = kmedoids.fit_predict(
                    c_vec[:, concept_id].reshape(-1, 1)
                )
            else:
                c_cluster_labels = kmedoids.fit_predict(c_vec[:, concept_id])

            # compute alignment with ground truth labels
            concept_homogeneity.append(
                homogeneity_score(c_test[:, concept_id], c_cluster_labels)
            )
            task_homogeneity.append(
                homogeneity_score(y_test, c_cluster_labels)
            )

        # compute the area under the curve
        concept_auc.append(np.trapz(np.array(concept_homogeneity)) / max_auc)
        task_auc.append(np.trapz(np.array(task_homogeneity)) / max_auc)

    # return the average alignment across all concepts
    concept_auc = np.mean(concept_auc)
    task_auc = np.mean(task_auc)
    if force_alignment:
        return concept_auc, task_auc, alignment
    return concept_auc, task_auc
            