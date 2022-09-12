import logging
import sklearn
import numpy as np
import metrics
import os
import joblib

import xgboost as xgb
import lightgbm as lgb
import training.utils as utils
import tensorflow as tf

from sklearn.model_selection import train_test_split



############################################
## XGBoost Training
############################################

def train_xgboost(
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
):
    utils.restart_seeds(seed)
    end_results = trial_results if trial_results is not None else {}
    old_results = (old_results or {}) if load_from_cache else {}
    verbosity = experiment_config.get("verbosity", 0)
    
    
    model_path = os.path.join(
        experiment_config["results_dir"],
        f"models/weights{extra_name}.model"
    )
    
    
    params = dict(
        max_depth=experiment_config.get('max_depth', 6),
        eta=experiment_config.get('learning_rate', 0.3),
        objective=(
            'multi:softprob'
            if experiment_config["num_outputs"] > 1 else
            'binary:logistic'
        ),
        num_class=len(np.unique(y_train)),
        nthread=experiment_config.get('nthread', 4),
        eval_metric=['merror'],
        seed=seed,
        gpu_id=0,
        tree_method='gpu_hist',
    )
    if experiment_config.get('patience', None) not in [None, 0, float("inf")]:
        params['num_early_stopping_rounds'] = experiment_config['patience']
    
    if load_from_cache and os.path.exists(model_path):
        logging.debug(prefix + "Found XGBoost model serialized! Loading it...")
        bst = xgb.Booster(params)
        bst.load_model(model_path)
        xgboost_time_trained = old_results.get('time_trained')
        xgboost_epochs_trained = old_results.get('epochs_trained')
    else:
        # Train it from scratch
        logging.info(prefix + "XGBoost model training...")
        if experiment_config.get("holdout_fraction", 0):
            if (c_train is not None) and (c_test is not None):
                x_train, x_val, y_train, y_val, c_train, c_val = train_test_split(
                    x_train,
                    y_train,
                    c_train,
                    test_size=experiment_config["holdout_fraction"],
                    random_state=42,
                )
            else:
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train,
                    y_train,
                    test_size=experiment_config["holdout_fraction"],
                    random_state=42,
                )
            dval = xgb.DMatrix(x_val, label=y_val)
            dtrain = xgb.DMatrix(x_train, label=y_train)
            evallist = [(dtrain, 'train'), (dval, 'val')]
        else:
            # Else we perform no validation
            dtrain = xgb.DMatrix(x_train, label=y_train)
            evallist = [(dtrain, 'train')]
        hist = {}
        bst, xgboost_time_trained = utils.timeit(
            xgb.train,
            params=params,
            dtrain=dtrain,
            num_boost_round=experiment_config['max_epochs'],
            evals=evallist,
            verbose_eval=verbosity > 0,
            evals_result=hist,
        )
        logging.debug(prefix + "\tXGBoost training completed")
        xgboost_epochs_trained = len(hist['train']['merror'])
        
        if experiment_config.get('save_history', True):
            joblib.dump(
                hist,
                os.path.join(
                    experiment_config["results_dir"],
                    "history",
                    (
                        f"train{extra_name}_hist.joblib"
                    )
                ),
            )
        
        logging.debug(prefix + "\tSerializing model")
        bst.save_model(model_path)
    
    # Log training times and whatnot
    if xgboost_epochs_trained is not None:
        end_results['epochs_trained'] = xgboost_epochs_trained
    if xgboost_time_trained is not None:
        end_results['time_trained'] = xgboost_time_trained

    logging.info(prefix + "\tEvaluating XGBoost..")
    dtest = xgb.DMatrix(x_test, label=y_test)
    test_output = bst.predict(dtest)
    end_results['acc'] = sklearn.metrics.accuracy_score(
        y_test,
        np.argmax(test_output, axis=-1),
    )
    end_results['auc'] = sklearn.metrics.roc_auc_score(
        tf.keras.utils.to_categorical(y_test),
        test_output,
        multi_class='ovo',
    )
    
    if (ground_truth_concept_masks is not None) and (c_train is not None) and (
        c_test is not None
    ):
        for method in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
            global_mask_dict = bst.get_score(fmap='', importance_type=method)
            global_mask = np.zeros(x_test.shape[-1], dtype=np.float32)
            for key, val in global_mask_dict.items():
                global_mask[int(key[1:])] = val
            # Normalize the mask
            global_mask_norm = global_mask / (np.max(global_mask) + 1e-10)
            logging.debug(prefix + f"\t\tPredicting feature importance with method {method}...")
            end_results[f'feat_importance_{method}_diff'] = metrics.feature_importance_diff(
                importance_masks=global_mask_norm,
                c_train=c_train,
                ground_truth_concept_masks=ground_truth_concept_masks,
            )
            if method == 'weight':
                end_results[f'feat_importance_diff'] = end_results[f'feat_importance_{method}_diff']
            logging.debug(prefix + f"\t\t\tDone: {end_results[f'feat_importance_{method}_diff']:.5f}")
            logging.debug(prefix + f"\t\tPredicting feature selection with method {method}...")
            end_results[f'feat_selection_{method}'] = metrics.feature_selection(
                importance_masks=global_mask,
                c_train=c_train,
                ground_truth_concept_masks=ground_truth_concept_masks,
            )
            if method == 'weight':
                end_results[f'feat_selection'] = end_results[f'feat_selection_{method}']
            logging.debug(prefix + f"\t\t\tDone: {end_results[f'feat_selection_{method}']:.5f}")
    
    if return_model:
        return end_results, bst
    return end_results

############################################
## LightGBM Training
############################################

def train_lightgbm(
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
):
    utils.restart_seeds(seed)
    end_results = trial_results if trial_results is not None else {}
    old_results = (old_results or {}) if load_from_cache else {}
    verbosity = experiment_config.get("verbosity", 0)
    
    
    model_path = os.path.join(
        experiment_config["results_dir"],
        f"models/weights{extra_name}.model"
    )
    
    
    params = dict(
        max_depth=experiment_config.get('max_depth', 6),
        num_leaves=experiment_config.get('num_leaves', 31),
        learning_rate=experiment_config.get('learning_rate', 0.1),
        objective=(
            'multiclass'
            if experiment_config["num_outputs"] > 1 else
            'binary'
        ),
        num_class=len(np.unique(y_train)),
        nthread=experiment_config.get('nthread', 4),
        metric=(
            'multiclass'
            if experiment_config["num_outputs"] > 1 else
            'auc_mu'
        ),
            
        seed=seed,
        num_iterations=experiment_config['max_epochs'],
        verbosity=verbosity,
        early_stopping_round=experiment_config['patience'],
#         device='gpu',
#         gpu_platform_id=0,
#         gpu_device_id=0,
    )
    
    if load_from_cache and os.path.exists(model_path):
        logging.debug(prefix + "Found LightGBM model serialized! Loading it...")
        bst = lgb.Booster(params, model_file=model_path)
        time_trained = old_results.get('time_trained')
        epochs_trained = old_results.get('epochs_trained')
    else:
        # Train it from scratch
        logging.info(prefix + "LightGBM model training...")
        if experiment_config.get("holdout_fraction", 0):
            if (c_train is not None) and (c_test is not None):
                x_train, x_val, y_train, y_val, c_train, c_val = train_test_split(
                    x_train,
                    y_train,
                    c_train,
                    test_size=experiment_config["holdout_fraction"],
                    random_state=42,
                )
            else:
                x_train, x_val, y_train, y_val = train_test_split(
                    x_train,
                    y_train,
                    test_size=experiment_config["holdout_fraction"],
                    random_state=42,
                )
            dval = lgb.Dataset(x_val, label=y_val)
            dtrain = lgb.Dataset(x_train, label=y_train)
            valid_sets = [dval]
        else:
            # Else we perform no validation
            dtrain = lgb.Dataset(x_train, label=y_train)
            valid_sets = []
        callbacks = []
        if experiment_config.get('patience', None) not in [None, 0, float("inf")]:
            callbacks = [lgb.early_stopping(stopping_rounds=experiment_config['patience'])]
        bst, time_trained = utils.timeit(
            lgb.train,
            params=params,
            train_set=dtrain,
            num_boost_round=experiment_config['max_epochs'],
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        logging.debug(prefix + "\tLightGBM training completed")
        if experiment_config.get('patience', None) not in [None, 0, float("inf")]:
            epochs_trained = bst.best_iteration
        else:
            epochs_trained = experiment_config['max_epochs']
        
        logging.debug(prefix + "\tSerializing model")
        if experiment_config.get('patience', None) not in [None, 0, float("inf")]:
            bst.save_model(model_path, num_iteration=bst.best_iteration)
        else:
            bst.save_model(model_path)
    
    # Log training times and whatnot
    if epochs_trained is not None:
        end_results['epochs_trained'] = epochs_trained
    if time_trained is not None:
        end_results['time_trained'] = time_trained

    logging.info(prefix + "\tEvaluating LightGBM..")
    if experiment_config.get('patience', None) not in [None, 0, float("inf")]:
        test_output = bst.predict(x_test, num_iteration=bst.best_iteration)
    else:
        test_output = bst.predict(x_test)
    end_results['acc'] = sklearn.metrics.accuracy_score(
        y_test,
        np.argmax(test_output, axis=-1),
    )
    end_results['auc'] = sklearn.metrics.roc_auc_score(
        tf.keras.utils.to_categorical(y_test),
        test_output,
        multi_class='ovo',
    )
    
    if (ground_truth_concept_masks is not None) and (c_train is not None) and (
        c_test is not None
    ):
        
        for method in ["split", "gain"]:
            if experiment_config.get('patience', None) not in [None, 0, float("inf")]:
                global_mask = bst.feature_importance(
                    importance_type=method,
                    iteration=bst.best_iteration,
                )
            else:
                global_mask = bst.feature_importance(importance_type=method)
            # Normalize the mask
            global_mask_norm = global_mask / (np.max(global_mask) + 1e-10)
            logging.debug(prefix + f"\t\tPredicting feature importance for method {method}...")
            end_results[f'feat_importance_{method}_diff'] = metrics.feature_importance_diff(
                importance_masks=global_mask_norm,
                c_train=c_train,
                ground_truth_concept_masks=ground_truth_concept_masks,
            )
            if method == 'split':
                end_results[f'feat_importance_diff'] = end_results[f'feat_importance_{method}_diff']
            logging.debug(prefix + f"\t\t\tDone: {end_results[f'feat_importance_{method}_diff']:.5f}")

            logging.debug(prefix + f"\t\tPredicting feature selection AUC for method {method}...")
            end_results[f'feat_selection_{method}'] = metrics.feature_selection(
                importance_masks=global_mask,
                c_train=c_train,
                ground_truth_concept_masks=ground_truth_concept_masks,
            )
            if method == 'split':
                end_results[f'feat_selection'] = end_results[f'feat_selection_{method}']
            logging.debug(prefix + f"\t\t\tDone: {end_results[f'feat_selection_{method}']:.5f}")
    if return_model:
        return end_results, bst
    return end_results
