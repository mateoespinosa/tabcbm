import logging
import numpy as np
import os

from sklearn.decomposition import PCA

import tabcbm.training.utils as utils
import tabcbm.training.representation_evaluation as representation_evaluation




############################################
## XGBoost Training
############################################

def train_pca(
    experiment_config,
    x_train,
    y_train,
    c_train,
    x_test,
    y_test,
    c_test,
    load_from_cache=False,
    seed=0,
    extra_name='',
    old_results=None,
    prefix="",
    ground_truth_concept_masks=None,
    trial_results=None,
    return_model=False,
    cat_feat_inds=None,
    cat_dims=None,
):
    utils.restart_seeds(seed)
    end_results = trial_results if trial_results is not None else {}
    old_results = (old_results or {}) if load_from_cache else {}

    pca_model = PCA(n_components=experiment_config['n_concepts'])
    _, pca_time_trained = utils.timeit(
        pca_model.fit,
        X=x_train,
    )
    logging.debug(prefix + "\tPCA training completed")

    if pca_time_trained is not None:
        end_results['time_trained'] = pca_time_trained

    logging.info(prefix + "\tEvaluating PCA..")
    np.save(
        os.path.join(
            experiment_config["results_dir"],
            f"explained_variance_ratio{extra_name}"
        ),
        pca_model.explained_variance_ratio_,
    )
    np.save(
        os.path.join(
            experiment_config["results_dir"],
            f"singular_values{extra_name}"
        ),
        pca_model.singular_values_,
    )
    test_concept_scores = pca_model.transform(x_test)
    if c_test is not None:
        representation_evaluation.evaluate_concept_representations(
            end_results=end_results,
            experiment_config=experiment_config,
            test_concept_scores=test_concept_scores,
            c_test=c_test,
            y_test=y_test,
            old_results=old_results,
            load_from_cache=load_from_cache,
            prefix=prefix,
        )

    if return_model:
        return end_results, pca_model
    return end_results