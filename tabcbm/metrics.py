import numpy as np
import scipy
import sklearn

from sklearn_extra.cluster import KMedoids
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import homogeneity_score
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sympy.utilities.iterables import multiset_permutations

import tabcbm.concepts_xai.evaluation.metrics.purity as purity

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
    threshold_ratio=list(np.arange(0.0, 1.0, 0.025)),
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
        if len(concept_importance_masks) < len(ground_truth_concept_masks):
            reference_masks = ground_truth_concept_masks[permutation, :]
            current_masks = concept_importance_masks
        else:
            reference_masks = ground_truth_concept_masks
            current_masks = concept_importance_masks[permutation, :]
        current_aucs = []
        for i in range(min(
            ground_truth_concept_masks.shape[0],
            concept_importance_masks.shape[0],
        )):
            mask = current_masks[i, :]
            if thresh is not None:
                mask = (mask >= thresh).astype(np.int32)
            ground_truth = reference_masks[i, :]
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
        if concept_scores.shape[-1] < c_test.shape[-1]:
            test_scores = c_test[:, list(filter(lambda x: x is not None, permutation))]
            scores = concept_scores
        else:
            test_scores = c_test
            scores = concept_scores[:, permutation]
        current_aucs = []
        for i in range(min(
            test_scores.shape[-1],
            concept_scores.shape[-1],
        )):
            # We take the max as the concept may be flipped
            current_score = max(
                sklearn.metrics.roc_auc_score(
                    test_scores[:, i],
                    scores[:, i],
                ),
                sklearn.metrics.roc_auc_score(
                    test_scores[:, i],
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
    n_ground_truth = c_train.shape[-1]
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
        if c_vec.shape[-1] < c_test.shape[-1]:
            # Then the alignment will need to be done backwards as
            # we will have to get rid of the dimensions in c_test
            # which have no aligment at all
            c_test = c_test[:, list(filter(lambda x: x is not None, alignment))]
        else:
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


##############
## R4
##############

###############################################################################
#
# R4 and R4c scores (our contribution)
#
# These metrics quantify the extent to which every dimension of a ground-truth
# representation V can be mapped individually (via an invertible function) to
# dimensions of a learned representation Z. They accomplish this by considering
# the R^2 coefficient of determination in both directions and taking geometric
# means.
#
# The conditional version (R4c) also takes into account the hierarchy, scoping
# comparisons to cases where both learned and ground-truth factors are active,
# and not penalizing minor differences in the distribution of continuous dims.
#
###############################################################################

def activity_mask(v):
    # Slight kludge to detect activity; could pass a separate mask variable
    # instead
    return (np.abs(v) > 1e-10).astype(int)

def is_categorical(v, max_uniq=10):
    # Also kind of a kludge, but assume a variable is categorical if it's
    # integer-valued and there are few possible options. Could use the
    # hierarchy object instead.
    return len(np.unique(v)) <= max_uniq and np.allclose(v.astype(int), v)

def sample_R2_oneway(inputs, targets, reg=GradientBoostingRegressor, kls=GradientBoostingClassifier):
    if len(inputs) < 2:
        # Handle edge case of nearly empty input
        return 0

    x_train, x_test, y_train, y_test = train_test_split(inputs.reshape(-1,1), targets)
    n_uniq = min(len(np.unique(y_train)), len(np.unique(y_test)))

    if n_uniq == 1:
        # Handle edge case of only one target
        return 1
    elif is_categorical(targets):
        # Use a classifier for categorical data
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        model = kls()
    else:
        # Use a regressor otherwise
        model = reg()

    # Return the R^2 (or accuracy) score
    return model.fit(x_train, y_train).score(x_test, y_test)

def R2_oneway(inputs, targets, iters=5, **kw):
    # Repeatedly compute R^2 over random splits
    return np.mean([sample_R2_oneway(inputs, targets, **kw) for _ in range(iters)])

def R2_bothways(x, y):
    # Take the geometric mean of R^2 in both directions
    r1 = max(0, R2_oneway(x,y))
    r2 = max(0, R2_oneway(y,x))
    return np.sqrt(r1*r2)

def R4_scores(V, Z):
    # For each dimension, find the best R2_bothways
    scores = []

    for i in range(V.shape[1]):
        best = 0
        for j in range(Z.shape[1]):
            best = max(best, R2_bothways(V[:,i], Z[:,j]))
        scores.append(best)

    return np.mean(scores)


###############################################################################
#
# Mutual Information Gap (MIG) Baseline
#
# Technically not defined for continuous targets, but we discretize with 20-bin
# histograms.
#
###############################################################################

def estimate_mutual_information(X, Y, bins=20):
  hist = np.histogram2d(X, Y, bins)[0] # approximate joint
  info = mutual_info_score(None, None, contingency=hist)
  return info / np.log(2) # bits

def estimate_entropy(X, **kw):
  return estimate_mutual_information(X, X, **kw)

def MIG(Z_true, Z_learned, **kw):
    K = Z_true.shape[1]
    gap = 0
    for k in range(K):
        H = estimate_entropy(Z_true[:,k], **kw)
        MIs = sorted([
          estimate_mutual_information(Z_learned[:,j], Z_true[:,k], **kw)
          for j in range(Z_learned.shape[1])
        ], reverse=True)
        if len(MIs) > 1:
            gap += (MIs[0] - MIs[1]) / (H * K)
        else:
            gap += MIs[0] / (H * K)
    return gap

###############################################################################
#
# SAP Score Baseline
#
###############################################################################

def SAP(V, Z):
    saps = []

    for i in range(V.shape[1]):
        v = V[:,i]

        scores = []
        for j in range(Z.shape[1]):
            if is_categorical(v):
                model = LinearSVC(C=0.01, class_weight="balanced")
                v = v.astype(int)
            else:
                model = LinearRegression()
            z = Z[:,j].reshape(-1,1)
            scores.append(model.fit(z,v).score(z,v))
        scores = list(sorted(scores))
        if len(scores) > 1:
            saps.append(scores[-1] - scores[-2])
        else:
            saps.append(scores[-1])

    return np.mean(saps)

###############################################################################
#
# DCI (Disentanglement, Completeness, Informativeness) Baseline
#
# Code adapted from https://github.com/google-research/disentanglement_lib,
# original paper at https://openreview.net/forum?id=By-7dz-AZ.
#
###############################################################################

def DCI(gen_factors, latents):
  """Computes score based on both training and testing codes and factors."""
  mus_train, mus_test, ys_train, ys_test = train_test_split(gen_factors, latents, test_size=0.1)
  scores = {}
  importance_matrix, train_err, test_err = compute_importance_gbt(mus_train, ys_train, mus_test, ys_test)
  assert importance_matrix.shape[0] == mus_train.shape[1]
  assert importance_matrix.shape[1] == ys_train.shape[1]
  scores["informativeness_train"] = train_err
  scores["informativeness_test"] = test_err
  scores["disentanglement"] = disentanglement(importance_matrix) if latents.shape[-1] > 1 else 1
  scores["completeness"] = completeness(importance_matrix)
  return scores["disentanglement"], scores["completeness"], scores["informativeness_test"]

def compute_importance_gbt(x_train, y_train, x_test, y_test):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[1]
  num_codes = x_train.shape[1]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = GradientBoostingRegressor()
    model.fit(x_train, y_train[:,i])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(model.score(x_train, y_train[:,i]))
    test_loss.append(model.score(x_test, y_test[:,i]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / (importance_matrix.sum() + 1e-12)
    return np.sum(per_code*code_importance)

def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)

###############################################################################
#
# FactorVAE Score Baseline
#
# Code adapted from https://github.com/google-research/disentanglement_lib,
# original paper at https://arxiv.org/abs/1802.05983
#
###############################################################################

def FactorVAE(ground_truth_X,
              ground_truth_Z,
              representation_function,
              random_state=np.random.RandomState(0),
              batch_size=64,
              num_train=4000,
              num_eval=2000,
              num_variance_estimate=1000):
  """Computes the FactorVAE disentanglement metric.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    batch_size: Number of points to be used to compute the training_sample.
    num_train: Number of points used for training.
    num_eval: Number of points used for evaluation.
    num_variance_estimate: Number of points used to estimate global variances.

  Returns:
    Dictionary with scores:
      train_accuracy: Accuracy on training set.
      eval_accuracy: Accuracy on evaluation set.
  """
  global_variances = _compute_variances(ground_truth_X,
                                        representation_function,
                                        num_variance_estimate, random_state)
  active_dims = _prune_dims(global_variances)
  scores_dict = {}

  if not active_dims.any():
    scores_dict["train_accuracy"] = 0.
    scores_dict["eval_accuracy"] = 0.
    scores_dict["num_active_dims"] = 0
    return scores_dict

  training_votes = _generate_training_batch(ground_truth_X, ground_truth_Z,
                                            representation_function, batch_size,
                                            num_train, random_state,
                                            global_variances, active_dims)
  classifier = np.argmax(training_votes, axis=0)
  other_index = np.arange(training_votes.shape[1])

  train_accuracy = np.sum(
      training_votes[classifier, other_index]) * 1. / np.sum(training_votes)

  eval_votes = _generate_training_batch(ground_truth_X, ground_truth_Z,
                                        representation_function, batch_size,
                                        num_eval, random_state,
                                        global_variances, active_dims)

  eval_accuracy = np.sum(eval_votes[classifier,
                                    other_index]) * 1. / np.sum(eval_votes)
  scores_dict["train_accuracy"] = train_accuracy
  scores_dict["eval_accuracy"] = eval_accuracy
  scores_dict["num_active_dims"] = len(active_dims)
  return scores_dict["eval_accuracy"]

def obtain_representation(observations, representation_function, batch_size):
  """"Obtain representations from observations.
  Args:
    observations: Observations for which we compute the representation.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Batch size to compute the representation.
  Returns:
    representations: Codes (num_codes, num_points)-Numpy array.
  """
  representations = None
  num_points = observations.shape[0]
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_observations = observations[i:i + num_points_iter]
    if i == 0:
      representations = representation_function(current_observations)
    else:
      representations = np.vstack((representations,
                                   representation_function(
                                       current_observations)))
    i += num_points_iter
  return np.transpose(representations)

def _prune_dims(variances, threshold=0.05):
  """Mask for dimensions collapsed to the prior."""
  scale_z = np.sqrt(variances)
  return scale_z >= threshold


def _compute_variances(ground_truth_X,
                       representation_function,
                       batch_size,
                       random_state,
                       eval_batch_size=64):
  """Computes the variance for each dimension of the representation.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the variances.
    random_state: Numpy random state used for randomness.
    eval_batch_size: Batch size used to eval representation.

  Returns:
    Vector with the variance of each dimension.
  """
  observation_indexes = np.arange(len(ground_truth_X))
  np.random.shuffle(observation_indexes)
  observations = ground_truth_X[observation_indexes][:batch_size]
  representations = obtain_representation(observations,
                                                representation_function,
                                                eval_batch_size)
  representations = np.transpose(representations)
  assert representations.shape[0] == batch_size
  return np.var(representations, axis=0, ddof=1)

def _generate_training_sample(ground_truth_X, ground_truth_Z, representation_function,
                              batch_size, random_state, global_variances,
                              active_dims, tol=0.001):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the training_sample.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

  Returns:
    factor_index: Index of factor coordinate to be used.
    argmin: Index of representation coordinate with the least variance.
  """
  # Select random coordinate to keep fixed.
  factor_index = random_state.randint(ground_truth_Z.shape[1])

  # Pick fixed factor value
  factor_value = np.random.choice(ground_truth_Z[:,factor_index])

  # Find indices of examples with closest values
  factor_diffs = np.abs(ground_truth_Z[:,factor_index]-factor_value)
  sorted_observation_indexes = factor_diffs.argsort()
  exact_observation_indexes = np.argwhere(factor_diffs == 0)[:,0]
  np.random.shuffle(exact_observation_indexes)
  if len(exact_observation_indexes) >= batch_size:
      # If there are enough which are exactly equal, shuffle
      observation_indexes = exact_observation_indexes[:batch_size]
  else:
      # If not, just pick all of the closest
      observation_indexes = sorted_observation_indexes[:batch_size]

  # Obtain the observations.
  observations = ground_truth_X[observation_indexes]

  representations = representation_function(observations)
  local_variances = np.var(representations, axis=0, ddof=1)
  argmin = np.argmin(local_variances[active_dims] /
                     global_variances[active_dims])
  return factor_index, argmin


def _generate_training_batch(ground_truth_X, ground_truth_Z, representation_function,
                             batch_size, num_points, random_state,
                             global_variances, active_dims):
  """Sample a set of training samples based on a batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Number of points to be used to compute the training_sample.
    num_points: Number of points to be sampled for training set.
    random_state: Numpy random state used for randomness.
    global_variances: Numpy vector with variances for all dimensions of
      representation.
    active_dims: Indexes of active dimensions.

  Returns:
    (num_factors, dim_representation)-sized numpy array with votes.
  """
  votes = np.zeros((ground_truth_Z.shape[1], global_variances.shape[0]),
                   dtype=np.int64)
  for _ in range(num_points):
    factor_index, argmin = _generate_training_sample(ground_truth_X, ground_truth_Z,
                                                     representation_function,
                                                     batch_size, random_state,
                                                     global_variances,
                                                     active_dims)
    votes[factor_index, argmin] += 1
  return votes
