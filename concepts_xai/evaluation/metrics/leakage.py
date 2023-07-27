"""
Taken from https://github.com/dmitrykazhdan/concept-based-xai.

All credit due to Kazhdan et al. (arXiv:2104.06917).
"""

'''
Implementation of several measures of leakage in a given concept-decomposable
model.
'''
import numpy as np
import sklearn


def compute_concept_aucs(
    cw_model,
    encoder,
    cw_layer,
    x_test,
    c_test,
    num_concepts,
    aggregator='max_pool_mean',
):
    concept_scores = cw_model.layers[cw_layer].concept_scores(
        encoder(x_test),
        aggregator=aggregator,
    ).numpy()[:, list(range(num_concepts))]

    return np.array([
        sklearn.metrics.roc_auc_score(c_test[:, i] == 1, concept_scores[:, i])
        for i in range(num_concepts)
    ])
