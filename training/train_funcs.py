import time
import sklearn
import scipy
import tensorflow as tf
import numpy as np
from collections import defaultdict
import metrics
import concepts_xai.methods.OCACE.topicModel as CCD
import concepts_xai.evaluation.metrics.completeness as completeness
import os
from pathlib import Path
import models.models as models
import joblib
import random
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
import xgboost as xgb
import concepts_xai.methods.SENN.aggregators as aggregators
import concepts_xai.methods.VAE.betaVAE as beta_vae
import concepts_xai.methods.VAE.losses as vae_losses
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
import torch
import shutil
import zipfile
from models.tabcbm import TabCBM
import copy
import pandas as pd
import lightgbm as lgb
from numba import cuda 
import io
from keras import backend as K
import multiprocessing
import warnings
import gc
import logging
import nvidia_smi


############################################
## Utils
############################################

def _evaluate_expressions(config):
    for key, val in config.items():
        if isinstance(val, (str,)) and len(val) >= 4 and (
            val[0:2] == "{{" and val[-2:] == "}}"
        ):
            # Then do a simple substitution here
            config[key] = val[1:-1].format(**config)
            # And then try and convert it into its
            # possibly numerical value. We first try
            # ints then floats
            try:
                config[key] = int(config[key])
            except:
                try:
                    config[key] = float(config[key])
                except:
                    pass
           
        
def posible_load(
    key,
    run_fn,
    old_results,
    load_from_cache=True,
):
    keys = key
    if not isinstance(keys, (list, tuple)):
        keys = [key]
        
    if old_results and load_from_cache:
        result = []
        for k in keys:
            if k in old_results:
                result.append(old_results[k])
            else:
                break
        if len(result) == len(keys):
            return result[0] if len(keys) == 1 else tuple(result)
    return run_fn()

def timeit(f, *args, **kwargs):
    start = time.time()
    result = f(*args, **kwargs)
    end = time.time()
    return result, (end - start)

def restart_seeds(trial=0):
    os.environ['PYTHONHASHSEED'] = str(42 + trial)
    tf.random.set_seed(42 + trial)
    np.random.seed(42 + trial)
    random.seed(42 + trial)
    # And also let's reduce the noise from warnings
    warnings.filterwarnings('ignore')
    
    # Reset the logging in case we are using a subprocess
    print("\tSetting log level to:", os.environ.get('LOGLEVEL', 'WARNING').upper())
    logging.getLogger().setLevel(
        os.environ.get('LOGLEVEL', 'WARNING').upper()
    )

def get_argmax_concept_explanations(preds, class_theta_scores):
    if len(preds.shape) == 2:
        inds = np.argmax(preds, axis=-1)
    else:
        inds = (preds >= 0.5).astype(np.int32)
    result = np.take_along_axis(
        class_theta_scores,
        np.expand_dims(np.expand_dims(inds, axis=-1), axis=-1),
        axis=1,
    )
    return np.squeeze(result, axis=1)

def save_tabnet_model(self, path):
    """Saving TabNet model in two distinct files.
    Parameters
    ----------
    path : str
        Path of the model.
    Returns
    -------
    str
        input filepath with ".zip" appended
    """
    saved_params = {}
    init_params = {}
    for key, val in self.get_params().items():
        if isinstance(val, type):
            # Don't save torch specific params
            continue
        else:
            init_params[key] = val
    saved_params["init_params"] = init_params

    class_attrs = {
        "preds_mapper": self.preds_mapper
    }
    saved_params["class_attrs"] = class_attrs

    # Create folder
    Path(path).mkdir(parents=True, exist_ok=True)

    # Save models params
    joblib.dump(saved_params, Path(path).joinpath("model_params.joblib"))

    # Save state_dict
    torch.save(self.network.state_dict(), Path(path).joinpath("network.pt"))
    shutil.make_archive(path, "zip", path)
    shutil.rmtree(path)
    return f"{path}.zip"

def load_tabnet_model(self, filepath):
    """Load TabNet model.
    Parameters
    ----------
    filepath : str
        Path of the model.
    """
    try:
        with zipfile.ZipFile(filepath) as z:
            with z.open("model_params.joblib") as f:
                loaded_params = joblib.load(f)
            loaded_params["init_params"]["device_name"] = self.device_name
            with z.open("network.pt") as f:
                try:
                    saved_state_dict = torch.load(f, map_location=self.device)
                except io.UnsupportedOperation:
                    # In Python <3.7, the returned file object is not seekable (which at least
                    # some versions of PyTorch require) - so we'll try buffering it in to a
                    # BytesIO instead:
                    saved_state_dict = torch.load(
                        io.BytesIO(f.read()),
                        map_location=self.device,
                    )
    except KeyError:
        raise KeyError("Your zip file is missing at least one component")

    self.__init__(**loaded_params["init_params"])

    self._set_network()
    self.network.load_state_dict(saved_state_dict)
    self.network.eval()
    self.load_class_attrs(loaded_params["class_attrs"])

    return

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
    restart_seeds(seed)
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
        bst, xgboost_time_trained = timeit(
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
    restart_seeds(seed)
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
        bst, time_trained = timeit(
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

############################################
## MLP Training
############################################

def train_mlp(
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
    trial_results=None,
    return_model=False,
):
    restart_seeds(seed)
    end_results =  trial_results if trial_results is not None else {}
    old_results = (old_results or {}) if load_from_cache else {}
    verbosity = experiment_config.get("verbosity", 0)
    
    # Proceed to do and end-to-end model in case we want to
    # do some task-specific pretraining
    end_to_end_model, _, _ = models.construct_end_to_end_model(
        input_shape=experiment_config["input_shape"],
        num_outputs=experiment_config["num_outputs"],
        learning_rate=experiment_config["learning_rate"],
        encoder=models.construct_encoder(
            input_shape=experiment_config["input_shape"],
            latent_dims=experiment_config["latent_dims"],
            include_bn=experiment_config.get("include_bn", False),
            units=experiment_config["encoder_units"] + experiment_config.get('mlp_extra_units', []),
            latent_act=experiment_config.get("latent_act", None),
        ),
        decoder=models.construct_decoder(
            units=experiment_config["decoder_units"],
            num_outputs=experiment_config["num_outputs"],
        ),
    )
    
    end_to_end_model_path = os.path.join(
        experiment_config["results_dir"],
        f"models/model{extra_name}"
    )
    if  load_from_cache and os.path.exists(end_to_end_model_path):
        logging.debug(prefix + "Found MLP model serialized! Loading it up...")
        # Then time to load up the end-to-end model!
        end_to_end_model = tf.keras.models.load_model(end_to_end_model_path)
        end_to_end_epochs_trained = old_results.get('epochs_trained')
        end_to_end_time_trained = old_results.get('time_trained')
    else:
        logging.info(prefix + "MLP training...")
        early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
            monitor=experiment_config.get(
                "early_stop_metric",
                "val_loss",
            ),
            min_delta=experiment_config["min_delta"],
            patience=experiment_config["patience"],
            restore_best_weights=True,
            verbose=2,
            mode=experiment_config.get(
                "early_stop_mode",
                "min",
            ),
        )
        if experiment_config.get('save_history', True):
            callbacks = [
                early_stopping_monitor,                 
                tf.keras.callbacks.CSVLogger(
                    os.path.join(
                        experiment_config["results_dir"],
                        "history",
                        (
                            f"model{extra_name}_hist.csv"
                        )
                    ),
                    append=True
                ),
            ],
        else:
            callbacks = [early_stopping_monitor]
        end_to_end_hist, end_to_end_time_trained = timeit(
            end_to_end_model.fit,
            x=x_train,
            y=y_train,
            epochs=experiment_config["max_epochs"],
            batch_size=experiment_config["batch_size"],
            callbacks=callbacks,
            validation_split=experiment_config["holdout_fraction"],
            verbose=verbosity,
        )
        end_to_end_epochs_trained = len(end_to_end_hist.history['loss'])
        end_to_end_model.save(end_to_end_model_path)
        logging.debug(prefix + "\tMLP training completed")
        
    end_results['num_params'] = (
        np.sum([np.prod(p.shape) for p in end_to_end_model.trainable_weights])
    )
    logging.info(prefix + "\tEvaluating MLP model")
    if experiment_config["num_outputs"] > 1:
        preds = scipy.special.softmax(
            end_to_end_model.predict(x_test),
            axis=-1,
        )

        one_hot_labels = tf.keras.utils.to_categorical(y_test)
        end_results['acc'] = sklearn.metrics.accuracy_score(
            y_test,
            np.argmax(preds, axis=-1),
        )

        # And select just the labels that are in fact being used
        end_results['auc'] = sklearn.metrics.roc_auc_score(
            one_hot_labels,
            preds,
            multi_class='ovo',
        )
    else:
        end_results['acc'] = sklearn.metrics.accuracy_score(
            y_test,
            end_to_end_model.predict(x_test),
        )
        end_results['auc'] = sklearn.metrics.roc_auc_score(
            y_test,
            end_to_end_model.predict(x_test),
        )
    
    # Log training times and whatnot
    if end_to_end_epochs_trained is not None:
        end_results['epochs_trained'] = end_to_end_epochs_trained
    if end_to_end_time_trained is not None:
        end_results['time_trained'] = end_to_end_time_trained
    
    if return_model:
        return end_results, end_to_end_model
    return end_results

############################################
## TabNet Training
############################################

def train_tabnet(
    experiment_config,
    x_train,
    y_train,
    c_train,
    x_test,
    y_test,
    c_test,
    load_from_cache=False,
    seed=0,
    old_results=None,
    prefix="",
    extra_name='',
    cat_idxs=[],
    cat_dims=[],
    ground_truth_concept_masks=None,
    trial_results=None,
    return_model=False,
):
    restart_seeds(seed)
    end_results =  trial_results if trial_results is not None else {}
    old_results = (old_results or {}) if load_from_cache else {}
    verbosity = experiment_config.get("verbosity", 0)
    
    tabnet_params = dict(
        n_d=experiment_config.get('n_d', 8),
        n_a=experiment_config.get('n_a', 8),
        n_steps=experiment_config.get('n_steps', 3),
        gamma=experiment_config.get('gamma', 1.3),
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=experiment_config.get('cat_emb_dim', 1),
        n_independent=experiment_config.get('n_independent', 2),
        n_shared=experiment_config.get('n_shared', 2),
        epsilon=experiment_config.get('eps', 1e-15),
        seed=seed,
        momentum=experiment_config.get('momentum', 0.02),
        lambda_sparse=experiment_config.get('lambda_sparse', 1e-3),
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=experiment_config.get('initial_lr', 0.02)),
#         scheduler_params={
#             "step_size":experiment_config.get('decay_step_size', 10),
#              "gamma":experiment_config.get('decay_rate', 0.9),
#         },
#         scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=verbosity,
    )
    
    tabnet_path = os.path.join(
        experiment_config["results_dir"],
        f"models/tabnet{extra_name}"
    )
    if  load_from_cache and os.path.exists(tabnet_path + ".zip"):
        logging.debug(prefix + "Found TabNet model serialized! Loading it up...")
        # Then time to load up the end-to-end model!
        tabnet = TabNetClassifier(**tabnet_params)
        load_tabnet_model(tabnet, tabnet_path + ".zip")
        epochs_trained = old_results.get('epochs_trained')
        time_trained = old_results.get('time_trained')
        pretrain_epochs_trained = old_results.get('pretrain_epochs_trained')
        pretrain_time_trained = old_results.get('pretrain_time_trained')
    else:
        if experiment_config["holdout_fraction"]:
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
            eval_set = [(x_val, y_val)]
        else:
            eval_set = []
        
        extra_params = dict(
            batch_size=experiment_config["batch_size"],
            virtual_batch_size=experiment_config["virtual_batch_size"],
            patience=experiment_config.get("patience", float("inf")),
            num_workers=experiment_config.get('num_workers', 8),
        )
        if experiment_config.get('pretrain_epochs'):
            # TabNetPretrainer
            logging.info(prefix + "Self-supervised pre-training of TabNet...")
            pretrained_model = TabNetPretrainer(
                mask_type='entmax',
                **tabnet_params,
            )
            _, pretrain_time_trained = timeit(
                pretrained_model.fit,
                X_train=x_train,
                eval_set=[x_val],
                max_epochs=experiment_config['pretrain_epochs'],
                pretraining_ratio=experiment_config.get('pretraining_ratio', 0.8),
                pin_memory=False, # Setting this to false to avoid hoarding the GPU
                **extra_params,
            )
            end_results['pretrain_num_parameters'] = sum(
                [p.numel() for p in pretrained_model.network.parameters() if p.requires_grad]
            )
            
            if experiment_config.get("patience", float("inf")) not in [None, 0, float("inf")]:
                pretrain_epochs_trained = len(pretrained_model._callback_container.callbacks[0].history["loss"])
            else:
                pretrain_epochs_trained = experiment_config["pretrain_epochs"]
            
            extra_params["from_unsupervised"] = pretrained_model
            
            logging.debug(prefix + "\tDone!")
        else:
            pretrain_epochs_trained = None
            pretrain_time_trained = 0
        logging.info(prefix + "TabNet main model training...")
        
        tabnet = TabNetClassifier(**tabnet_params)
        _, time_trained = timeit(
            tabnet.fit,
            X_train=x_train,
            y_train=y_train,
            eval_set=eval_set,
            max_epochs=experiment_config["max_epochs"],
            pin_memory=False, # Setting this to false to avoid hoarding the GPU
            **extra_params,
        )
        end_results['num_parameters'] = sum(
            [p.numel() for p in tabnet.network.parameters() if p.requires_grad]
        )
        if experiment_config.get("patience", float("inf")) not in [None, 0, float("inf")]:
            epochs_trained = len(tabnet._callback_container.callbacks[0].history["loss"])
        else:
            epochs_trained = experiment_config["max_epochs"]
        save_tabnet_model(tabnet, tabnet_path)
        logging.debug(prefix + "\tDone!")
    
    if 'pretrain_num_parameters' not in end_results and load_from_cache and (
        'pretrain_num_parameters' in old_results
    ):
        end_results['pretrain_num_parameters'] = old_results['pretrain_num_parameters']
    if 'pretrain_num_parameters' in end_results:
        logging.debug(prefix + f"\tNumber of pretrain parameters = {end_results['pretrain_num_parameters']}")
    end_results['num_parameters'] = sum(
        [p.numel() for p in tabnet.network.parameters() if p.requires_grad]
    )
    logging.debug(prefix + f"\tNumber of TabNet parameters = {end_results['num_parameters']}")
    logging.info(prefix + "\tEvaluating TabNet model")
    preds = tabnet.predict(x_test)
    if experiment_config["num_outputs"] > 1:
        one_hot_labels = tf.keras.utils.to_categorical(y_test)
        end_results['acc'] = sklearn.metrics.accuracy_score(
            y_test,
            preds,
        )

        # And select just the labels that are in fact being used
        end_results['auc'] = sklearn.metrics.roc_auc_score(
            one_hot_labels,
            tf.one_hot(preds, experiment_config['num_outputs']),
            multi_class='ovo',
        )
    else:
        end_results['acc'] = sklearn.metrics.accuracy_score(
            y_test,
            preds,
        )
        end_results['auc'] = sklearn.metrics.roc_auc_score(
            y_test,
            preds,
        )
    
    # Evaluating TabNet's feature importance prediction
    if (ground_truth_concept_masks is not None) and (c_train is not None) and (
        c_test is not None
    ):
        global_masks, individual_masks = tabnet.explain(x_train)
        # Normalize them
        global_masks /= (np.sum(global_masks, axis=1)[:, None] + 1e-10)
        logging.debug(prefix + "\t\tPredicting feature importance matching...")
        end_results['feat_importance_diff'] = metrics.feature_importance_diff(
            importance_masks=global_masks,
            c_train=c_train,
            ground_truth_concept_masks=ground_truth_concept_masks,
        )
        logging.debug(prefix + f"\t\t\tDone: {end_results['feat_importance_diff']:.5f}")
        logging.debug(prefix + "\t\tPredicting feature selection matching...")
        end_results['feat_selection'] = metrics.feature_selection(
            importance_masks=global_masks,
            c_train=c_train,
            ground_truth_concept_masks=ground_truth_concept_masks,
        )
        logging.debug(prefix + f"\t\t\tDone: {end_results['feat_selection']:.5f}")
    # Log training times and whatnot
    if epochs_trained is not None:
        end_results['epochs_trained'] = epochs_trained
    if time_trained is not None:
        end_results['time_trained'] = time_trained
    
    if pretrain_epochs_trained is not None:
        end_results['pretrain_epochs_trained'] = pretrain_epochs_trained
    if pretrain_time_trained is not None:
        end_results['pretrain_time_trained'] = pretrain_time_trained
    if return_model:
        return end_results, tabnet
    del tabnet
    return end_results

############################################
## SENN Training
############################################

def train_senn(
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
    trial_results=None,
    return_model=False,
):
    restart_seeds(seed)
    end_results = trial_results if trial_results is not None else {}
    old_results = (old_results or {}) if load_from_cache else {}
    verbosity = experiment_config.get("verbosity", 0)
    
    
    # Proceed to do and end-to-end model in case we want to
    # do some task-specific pretraining
    concept_encoder, vae_encoder = models.construct_senn_encoder(
        input_shape=experiment_config["input_shape"],
        units=experiment_config["encoder_units"],
        latent_act=experiment_config.get("latent_act", None),
        latent_dims=experiment_config['n_concepts'],
    )
    concept_decoder = models.construct_vae_decoder(
        units=experiment_config["decoder_units"],
        output_shape=experiment_config["input_shape"][-1],
        latent_dims=experiment_config['n_concepts'],
    )
    coefficient_model = models.construct_senn_coefficient_model(
        units=experiment_config["coefficient_model_units"],
        num_concepts=experiment_config['n_concepts'],
        num_outputs=experiment_config["num_outputs"],
    )
    encoder_path = os.path.join(
        experiment_config["results_dir"],
        f"models/encoder{extra_name}"
    )
    if load_from_cache and os.path.exists(encoder_path):
        logging.debug(prefix + "Found cached SENN model! Loading it...")
        concept_encoder = tf.keras.models.load_model(encoder_path)
        concept_decoder = tf.keras.models.load_model(encoder_path.replace('/encoder_', '/decoder_'))
        coefficient_model = tf.keras.models.load_model(encoder_path.replace('/encoder_', '/coefficient_model_'))
        senn_model = models.construct_senn_model(
            concept_encoder=concept_encoder,
            concept_decoder=concept_decoder,
            coefficient_model=coefficient_model,
            num_outputs=experiment_config["num_outputs"],
            regularization_strength=experiment_config.get("regularization_strength", 0.1),
            learning_rate=experiment_config.get("learning_rate", 1e-3),
            sparsity_strength=experiment_config.get("sparsity_strength", 2e-5),
        )
        senn_epochs_trained = old_results.get('epochs_trained')
        senn_time_trained = old_results.get('time_trained')
        autoencoder_epochs_trained = old_results.get('autoencoder_epochs_trained')
        autoencoder_time_trained = old_results.get('autoencoder_time_trained')
    else:
        logging.info(prefix + "SENN Autoencoder model pre-training...")
        if experiment_config.get("pretrain_autoencoder_epochs"):
            def rec_loss_fn(y_true, y_pred):
                return tf.reduce_sum(
                    tf.square(y_true - y_pred),
                    [-1]
                )
            autoencoder = beta_vae.BetaVAE(
                encoder=vae_encoder,
                decoder=concept_decoder,
                loss_fn=rec_loss_fn,
                beta=experiment_config.get("beta", 1),
            )
            autoencoder.compile(
                optimizer=tf.keras.optimizers.Adam(
                    experiment_config.get("learning_rate", 1e-3)
                ),
            )
            if experiment_config.get('save_history', True):
                callbacks = [
                    tf.keras.callbacks.CSVLogger(
                        os.path.join(
                            experiment_config["results_dir"],
                            "history",
                            (
                                f"autoencoder_pretrain{extra_name}_hist.csv"
                            )
                        ),
                        append=True
                    ),
                ],
            else:
                callbacks = []
            autoencoder_hist, autoencoder_time_trained = timeit(
                autoencoder.fit,
                x=x_train,
                epochs=experiment_config["pretrain_autoencoder_epochs"],
                batch_size=experiment_config["batch_size"],
                validation_split=experiment_config["holdout_fraction"],
                verbose=verbosity,
                callbacks=callbacks,
            )
            autoencoder_epochs_trained = len(autoencoder_hist.history['loss'])
            logging.debug(prefix + "\tSENN autoencoder training completed")

        # Now time to actually construct and train the CBM
        logging.info(prefix + "Training SENN model...")
        senn_model = models.construct_senn_model(
            concept_encoder=concept_encoder,
            concept_decoder=concept_decoder,
            coefficient_model=coefficient_model,
            num_outputs=experiment_config["num_outputs"],
            regularization_strength=experiment_config.get("senn_regularization_strength", 0.1),
            learning_rate=experiment_config.get("learning_rate", 1e-3),
            sparsity_strength=experiment_config.get("senn_sparsity_strength", 2e-5),
        )

        early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
            monitor=experiment_config.get(
                "early_stop_metric",
                "val_loss",
            ),
            min_delta=experiment_config["min_delta"],
            patience=experiment_config["patience"],
            restore_best_weights=True,
            verbose=2,
            mode=experiment_config.get(
                "early_stop_mode",
                "max",
            ),
        )
        if experiment_config.get('save_history', True):
            callbacks = [
                early_stopping_monitor,                 
                tf.keras.callbacks.CSVLogger(
                    os.path.join(
                        experiment_config["results_dir"],
                        "history",
                        (
                            f"senn{extra_name}_hist.csv"
                        )
                    ),
                    append=True
                ),
            ],
        else:
            callbacks = [early_stopping_monitor]

        senn_hist, senn_time_trained = timeit(
            senn_model.fit,
            x=x_train,
            y=y_train,
            epochs=experiment_config["max_epochs"],
            batch_size=experiment_config["batch_size"],
            callbacks=callbacks,
            validation_split=experiment_config["holdout_fraction"],
            verbose=verbosity,
        )
        senn_epochs_trained = len(senn_hist.history['loss'])
        logging.debug(prefix + "\t\tDone!")
        logging.debug(prefix + "\tSerializing SENN model")
        concept_encoder.save(encoder_path)
        concept_decoder.save(encoder_path.replace('/encoder_', '/decoder_'))
        coefficient_model.save(encoder_path.replace('/encoder_', '/coefficient_model_'))
        logging.debug(prefix + "\t\tDone!")

    # Obtain SENN's predictions and scores for each of its discovered concepts
    end_results['num_params'] = (
        np.sum([np.prod(p.shape) for p in senn_model.trainable_weights])
    )
    logging.debug(prefix + f"\tNumber of SENN trainable parameters = {end_results['num_params']}")
    logging.info(prefix + "\tEvaluating SENN...")
    test_output, (_, x_test_theta_class_scores) = senn_model(x_test)
    test_output = test_output.numpy()
    test_concept_scores = get_argmax_concept_explanations(
        test_output,
        x_test_theta_class_scores.numpy(),
    )
    
    # Compute the model's accuracies
    logging.debug(prefix + "\t\tComputing accuracies...")
    if experiment_config["num_outputs"] > 1:
        # Then lets apply a softmax activation over all the probability
        # classes
        preds = scipy.special.softmax(
            test_output,
            axis=-1,
        )

        one_hot_labels = tf.keras.utils.to_categorical(y_test)
        end_results['acc'] = sklearn.metrics.accuracy_score(
            y_test,
            np.argmax(preds, axis=-1),
        )

        # And select just the labels that are in fact being used
        end_results['auc'] = sklearn.metrics.roc_auc_score(
            one_hot_labels,
            preds,
            multi_class='ovo',
        )
    else:
        end_results['acc'] = sklearn.metrics.accuracy_score(
            y_test,
            test_output,
        )
        end_results['auc'] = sklearn.metrics.roc_auc_score(
            y_test,
            test_output,
        )
    
    if c_test is not None:
        # Compute the CAS score
        logging.debug(prefix + "\t\tComputing CAS...")
        end_results['cas'], end_results['cas_task'], end_results['best_alignment'] = posible_load(
            key=['cas', 'cas_task', 'best_alignment'],
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.embedding_homogeneity(
                c_vec=test_concept_scores,
                c_test=c_test,
                y_test=y_test,
                step=experiment_config.get('cas_step', 2),
            )
        )
    logging.debug(prefix + "\t\tDone with evaluation...")
    if return_model:
        return end_results, senn_model
    return end_results
    
    
############################################
## CCD Training
############################################

def train_ccd(
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
    trial_results=None,
    return_model=False,
):
    restart_seeds(seed)
    channels_axis = (
        -1 if experiment_config.get("data_format", "channels_last") == "channels_last"
        else 1
    )
    
    end_results = trial_results if trial_results is not None else {}
    old_results = (old_results or {}) if load_from_cache else {}
    verbosity = experiment_config.get("verbosity", 0)
    
    if experiment_config["num_outputs"] <= 2:
        acc_fn = lambda y_true, y_pred: sklearn.metrics.roc_auc_score(
            y_true,
            y_pred
        )
    else:
        acc_fn = lambda y_true, y_pred: sklearn.metrics.roc_auc_score(
            tf.keras.utils.to_categorical(y_true),
            scipy.special.softmax(y_pred, axis=-1),
            multi_class='ovo',
        )
    
    # Proceed to do and end-to-end model in case we want to
    # do some task-specific pretraining
    end_to_end_model, encoder, decoder = models.construct_end_to_end_model(
        input_shape=experiment_config["input_shape"],
        num_outputs=experiment_config["num_outputs"],
        learning_rate=experiment_config["learning_rate"],
        encoder=models.construct_encoder(
            input_shape=experiment_config["input_shape"],
            latent_dims=experiment_config["latent_dims"],
            include_bn=experiment_config.get("include_bn", False),
            units=experiment_config["encoder_units"],
            latent_act=experiment_config.get("latent_act", None),
        ),
        decoder=models.construct_decoder(
            units=experiment_config["decoder_units"],
            num_outputs=experiment_config["num_outputs"],
        ),
    )
    
    encoder_path = os.path.join(
        experiment_config["results_dir"],
        f"models/encoder{extra_name}"
    )
    decoder_path = os.path.join(
        experiment_config["results_dir"],
        f"models/decoder{extra_name}"
    )
    if load_from_cache and os.path.exists(encoder_path):
        logging.debug(prefix + "Found encoder/decoder models serialized! We will unload them into the end-to-end model!")
        # Then time to load up the end-to-end model!
        encoder = tf.keras.models.load_model(encoder_path)
        decoder = tf.keras.models.load_model(decoder_path)
        end_to_end_model, encoder, decoder = models.construct_end_to_end_model(
            input_shape=experiment_config["input_shape"],
            num_outputs=experiment_config["num_outputs"],
            learning_rate=experiment_config["learning_rate"],
            encoder=encoder,
            decoder=decoder,
        )
        end_to_end_epochs_trained = old_results.get('pretrained_epochs_trained')
        end_to_end_time_trained = old_results.get('pretrained_time_trained')
    elif experiment_config.get("pretrain_epochs"):
        logging.info(prefix + "Model pre-training...")
        early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
            monitor=experiment_config.get(
                "early_stop_metric",
                "val_loss",
            ),
            min_delta=experiment_config["min_delta"],
            patience=experiment_config["patience"],
            restore_best_weights=True,
            verbose=2,
            mode=experiment_config.get(
                "early_stop_mode",
                "min",
            ),
        )
        if experiment_config.get('save_history', True):
            callbacks = [
                early_stopping_monitor,                 
                tf.keras.callbacks.CSVLogger(
                    os.path.join(
                        experiment_config["results_dir"],
                        "history",
                        (
                            f"end_to_end{extra_name}_hist.csv"
                        )
                    ),
                    append=True
                ),
            ],
        else:
            callbacks = [early_stopping_monitor]
        end_to_end_hist, end_to_end_time_trained = timeit(
            end_to_end_model.fit,
            x=x_train,
            y=y_train,
            epochs=experiment_config["pretrain_epochs"],
            batch_size=experiment_config["batch_size"],
            callbacks=callbacks,
            validation_split=experiment_config["holdout_fraction"],
            verbose=verbosity,
        )
        end_to_end_epochs_trained = len(end_to_end_hist.history['loss'])
        encoder.save(encoder_path)
        decoder.save(decoder_path)
        logging.debug(prefix + "\tModel pre-training completed")
            
    logging.info(prefix + "\tEvaluating end-to-end pretrained model")
    if experiment_config["num_outputs"] > 1:
        preds = scipy.special.softmax(
            end_to_end_model.predict(x_test),
            axis=-1,
        )

        one_hot_labels = tf.keras.utils.to_categorical(y_test)
        end_results['pre_train_acc'] = sklearn.metrics.accuracy_score(
            y_test,
            np.argmax(preds, axis=-1),
        )

        # And select just the labels that are in fact being used
        end_results['pre_train_auc'] = sklearn.metrics.roc_auc_score(
            one_hot_labels,
            preds,
            multi_class='ovo',
        )
    else:
        end_results['pre_train_acc'] = sklearn.metrics.accuracy_score(
            y_test,
            end_to_end_model.predict(x_test),
        )
        end_results['pre_train_auc'] = sklearn.metrics.roc_auc_score(
            y_test,
            end_to_end_model.predict(x_test),
        )
    logging.debug(prefix + f"\t\tPretrained model task accuracy: {end_results['pre_train_acc']*100:.2f}%")
    
    # Now extract our concept vectors
    topic_model = CCD.TopicModel(
        concepts_to_labels_model=decoder,
        n_channels=experiment_config["latent_dims"],
        n_concepts=experiment_config['n_concepts'],
        threshold=experiment_config.get("threshold", 0.5),
        loss_fn=end_to_end_model.loss,
        top_k=experiment_config.get("top_k", 32),
        lambda1=experiment_config.get("lambda1", 0.1),
        lambda2=experiment_config.get("lambda2", 0.1),
        seed=seed,
        eps=experiment_config.get("eps", 1e-5),
        data_format=experiment_config.get(
            "data_format",
            "channels_last"
        ),
        allow_gradient_flow_to_c2l=experiment_config.get(
            'allow_gradient_flow_to_c2l',
            False,
        ),
        acc_metric=(
            (
                lambda y_true, y_pred: tf.keras.metrics.sparse_categorical_accuracy(
                    y_true,
                    y_pred,
                )
            ) if experiment_config["num_outputs"] > 1 else tf.keras.metrics.binary_accuracy
        ),
    )
    topic_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            experiment_config.get("learning_rate", 1e-3),
        )
    )
    
    # See if we can load it
    ccd_g_model_path = os.path.join(
        experiment_config["results_dir"],
        f"models/ccd_g_model{extra_name}"
    )
    if load_from_cache and os.path.exists(ccd_g_model_path):
        logging.debug(prefix + "Found CCD's topic model saved! Let's properly unload it...")
        topic_model.g_model = tf.keras.models.load_model(ccd_g_model_path)
        topic_model.topic_vector.assign(np.load(
            ccd_g_model_path.replace('g_model', 'topic_vector_n') + ".npy"
        ))
        ccd_time_trained = old_results.get('time_trained')
        ccd_epochs_trained = old_results.get('epochs_trained')
    else:
        # Train it from scratch
        logging.info(prefix + "CCD's topic model training...")
        early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
            monitor=experiment_config.get(
                "early_stop_metric",
                "val_loss",
            ),
            min_delta=experiment_config["min_delta"],
            patience=experiment_config["patience"],
            restore_best_weights=True,
            verbose=2,
            mode=experiment_config.get(
                "early_stop_mode",
                "min",
            ),
        )
        if experiment_config.get('save_history', True):
            callbacks = [
                early_stopping_monitor,                 
                tf.keras.callbacks.CSVLogger(
                    os.path.join(
                        experiment_config["results_dir"],
                        "history",
                        (
                            f"topic_model{extra_name}_hist.csv"
                        )
                    ),
                    append=True
                ),
            ],
        else:
            callbacks = [early_stopping_monitor]
        ccd_hist, ccd_time_trained = timeit(
            topic_model.fit,
            x=encoder(x_train),
            y=y_train,
            callbacks=callbacks,
            epochs=experiment_config["max_epochs"],
            batch_size=experiment_config["batch_size"],
            validation_split=experiment_config["holdout_fraction"],
            verbose=verbosity,
        )
        logging.debug(prefix + "\tCCD's topic model training completed")
        ccd_epochs_trained = len(ccd_hist.history['loss'])
        logging.debug(prefix + "\tSerializing model")
        topic_model.g_model.save(ccd_g_model_path)
        np.save(
            ccd_g_model_path.replace('g_model', 'topic_vector_n') + ".npy",
            topic_model.topic_vector.numpy(),
        )
    end_results['pretrain_num_params'] = (
        np.sum([np.prod(p.shape) for p in end_to_end_model.trainable_weights])
    )
    logging.debug(prefix + f"\tNumber of pretrain model trainable parameters = {end_results['pretrain_num_params']}")
    end_results['num_params'] = (
        np.sum([np.prod(p.shape) for p in topic_model.trainable_weights])
    )
    logging.debug(prefix + f"\tNumber of TopicModel trainable parameters = {end_results['num_params']}")
    
    # Log training times and whatnot
    if end_to_end_epochs_trained is not None:
        end_results['pretrained_epochs_trained'] = end_to_end_epochs_trained
    if end_to_end_time_trained is not None:
        end_results['pretrained_time_trained'] = end_to_end_time_trained
    if ccd_epochs_trained is not None:
        end_results['epochs_trained'] = ccd_epochs_trained
    if ccd_time_trained is not None:
        end_results['time_trained'] = ccd_time_trained

    # Evaluate CCD's topic model
    logging.info(prefix + "\tEvaluating CCD's topic model")
    test_output, test_concept_scores = topic_model(encoder(x_test))
    test_concept_scores = test_concept_scores.numpy()
    test_output = test_output.numpy()
    if experiment_config["num_outputs"] > 1:
        # Then lets apply a softmax activation over all the probability
        # classes
        preds = scipy.special.softmax(
            test_output,
            axis=-1,
        )

        one_hot_labels = tf.keras.utils.to_categorical(y_test)
        end_results['acc'] = sklearn.metrics.accuracy_score(
            y_test,
            np.argmax(preds, axis=-1),
        )

        # And select just the labels that are in fact being used
        end_results['auc'] = sklearn.metrics.roc_auc_score(
            one_hot_labels,
            preds,
            multi_class='ovo',
        )
    else:
        end_results['acc'] = sklearn.metrics.accuracy_score(
            y_test,
            test_output,
        )
        end_results['auc'] = sklearn.metrics.roc_auc_score(
            y_test,
            test_output,
        )
    
    # Compute the CAS score
    if c_test is not None:
        end_results['cas'], end_results['cas_task'], end_results['best_alignment'] = posible_load(
            key=['cas', 'cas_task', 'best_alignment'],
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.embedding_homogeneity(
                c_vec=test_concept_scores,
                c_test=c_test,
                y_test=y_test,
                step=experiment_config.get('cas_step', 2),
            ),
        )
    
    # Let's see our topic model's completeness
    logging.debug(prefix + f"\t\tComputing CCD's completeness scores...")
    end_results['completeness'], _ = completeness.completeness_score(
        X=x_test,
        y=y_test,
        features_to_concepts_fn=encoder,
        concepts_to_labels_model=decoder,
        concept_vectors=np.transpose(topic_model.topic_vector.numpy()),
        task_loss=end_to_end_model.loss,
        channels_axis=channels_axis,
        concept_score_fn=lambda f, c: completeness.dot_prod_concept_score(
            features=f,
            concept_vectors=c,
            channels_axis=channels_axis,
            beta=experiment_config.get("threshold", 0.5),
        ),
        acc_fn=acc_fn,
        predictor_train_kwags={
            'epochs': experiment_config.get("completeness_epochs", 10),
            'batch_size': experiment_config["batch_size"],
            'verbose': 0,
        },
    )
    
    end_results['direct_completeness'], _ = completeness.direct_completeness_score(
        X=x_test,
        y=y_test,
        features_to_concepts_fn=encoder,
        concept_vectors=np.transpose(topic_model.topic_vector.numpy()),
        task_loss=end_to_end_model.loss,
        channels_axis=channels_axis,
        concept_score_fn=lambda f, c: completeness.dot_prod_concept_score(
            features=f,
            concept_vectors=c,
            channels_axis=channels_axis,
            beta=experiment_config.get("threshold", 0.5),
        ),
        acc_fn=acc_fn,
        predictor_train_kwags={
            'epochs': experiment_config.get("completeness_epochs", 10),
            'batch_size': experiment_config["batch_size"],
            'verbose': 0,
        },
    )
    
    if return_model:
        return end_results, topic_model
    return end_results

############################################
## TabCBM Training
############################################

def train_tabcbm(
    experiment_config,
    x_train,
    y_train,
    c_train,
    x_test,
    y_test,
    c_test,
    cov_mat=None,
    load_from_cache=False,
    ground_truth_concept_masks=None,
    extra_name="",
    seed=0,
    old_results=None,
    prefix="",
    trial_results=None,
    return_model=False,
):
    restart_seeds(seed)
    end_results = trial_results if trial_results is not None else {}
    old_results = (old_results or {}) if load_from_cache else {}
    verbosity = experiment_config.get("verbosity", 0)
    
    # Proceed to do and end-to-end model in case we want to
    # do some task-specific pretraining
    end_to_end_model, encoder, decoder = models.construct_end_to_end_model(
        input_shape=experiment_config["input_shape"],
        num_outputs=experiment_config["num_outputs"],
        learning_rate=experiment_config["learning_rate"],
        encoder=models.construct_encoder(
            input_shape=experiment_config["input_shape"],
            latent_dims=experiment_config["latent_dims"],
            include_bn=experiment_config.get("include_bn", False),
            units=experiment_config["encoder_units"],
            latent_act=experiment_config.get("latent_act", None),
        ),
        decoder=models.construct_decoder(
            units=experiment_config["decoder_units"],
            num_outputs=experiment_config["num_outputs"],
        ),
    )
    
    encoder_path = os.path.join(
        experiment_config["results_dir"],
        f"models/pretrained_encoder{extra_name}"
    )
    decoder_path = encoder_path.replace("pretrained_encoder", "pretrained_decoder")
    if  load_from_cache and os.path.exists(encoder_path):
        logging.debug(prefix + "Found encoder/decoder models serialized! We will unload them into the end-to-end model!")
        # Then time to load up the end-to-end model!
        encoder = tf.keras.models.load_model(encoder_path)
        decoder = tf.keras.models.load_model(decoder_path)
        end_to_end_model, encoder, decoder = models.construct_end_to_end_model(
            input_shape=experiment_config["input_shape"],
            num_outputs=experiment_config["num_outputs"],
            learning_rate=experiment_config["learning_rate"],
            encoder=encoder,
            decoder=decoder,
        )
    pretrained_epochs_trained = None
    pretrained_time_trained = None
    if experiment_config.get('pretrain_epochs') and (
        not (load_from_cache and os.path.exists(encoder_path))
    ):
            logging.info(prefix + "Model pre-training...")
            early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
                monitor=experiment_config.get(
                    "early_stop_metric_pretrain",
                    "val_loss",
                ),
                min_delta=experiment_config["min_delta"],
                patience=experiment_config["patience"],
                restore_best_weights=True,
                verbose=2,
                mode=experiment_config.get(
                    "early_stop_mode_pretrain",
                    "min",
                ),
            )
            if experiment_config.get('save_history', True):
                callbacks = [
                    early_stopping_monitor,                 
                    tf.keras.callbacks.CSVLogger(
                        os.path.join(
                            experiment_config["results_dir"],
                            "history",
                            (
                                f"pretrain{extra_name}_hist.csv"
                            )
                        ),
                        append=True
                    ),
                ],
            else:
                callbacks = [early_stopping_monitor]
            pretrain_hist, pretrain_time_trained = timeit(
                end_to_end_model.fit,
                x=x_train,
                y=y_train,
                epochs=experiment_config["pretrain_epochs"],
                batch_size=experiment_config["batch_size"],
                callbacks=callbacks,
                validation_split=experiment_config["holdout_fraction"],
                verbose=verbosity,
            )
            pretrain_epochs_trained = len(pretrain_hist.history['loss'])
            encoder.save(encoder_path)
            decoder.save(decoder_path)
            logging.debug(prefix + "\tModel pre-training completed")
    else:
        pretrain_epochs_trained = old_results.get('pretrained_epochs_trained')
        pretrain_time_trained = old_results.get('pretrained_time_trained')
        
    logging.info(prefix + "\tEvaluating end-to-end pretrained model")
    if experiment_config["num_outputs"] > 1:
        preds = scipy.special.softmax(
            end_to_end_model.predict(x_test),
            axis=-1,
        )

        one_hot_labels = tf.keras.utils.to_categorical(y_test)
        end_results['pre_train_acc'] = sklearn.metrics.accuracy_score(
            y_test,
            np.argmax(preds, axis=-1),
        )

        # And select just the labels that are in fact being used
        end_results['pre_train_auc'] = sklearn.metrics.roc_auc_score(
            one_hot_labels,
            preds,
            multi_class='ovo',
        )
    else:
        end_results['pre_train_acc'] = sklearn.metrics.accuracy_score(
            y_test,
            end_to_end_model.predict(x_test),
        )
        end_results['pre_train_auc'] = sklearn.metrics.roc_auc_score(
            y_test,
            end_to_end_model.predict(x_test),
        )
    logging.debug(prefix + f"\t\tPretrained model task accuracy: {end_results['pre_train_acc']*100:.2f}%")
    logging.debug(
        prefix +
        f"\t\tPretrained model params: {np.sum([np.prod(K.get_value(p).shape) for p in end_to_end_model.trainable_weights])}"
    )
    try:
        cov_mat = cov_mat if cov_mat is not None else np.corrcoef(x_train.T)
    except:
        cov_mat = None
    tab_cbm_params = dict(
        features_to_concepts_model=encoder,
        concepts_to_labels_model=decoder,
        latent_dims=experiment_config["latent_dims"],
        n_concepts=experiment_config['n_concepts'],
        n_supervised_concepts=experiment_config.get('n_supervised_concepts', 0),
        cov_mat=cov_mat,
        mean_inputs=(
            np.zeros(x_train.shape[1:], dtype=np.float32) if experiment_config["zero_mask"]
            else np.mean(x_train, axis=0)
        ),
        n_exclusive_concepts=experiment_config["n_exclusive_concepts"],
        gate_estimator_weight=experiment_config["gate_estimator_weight"],

        threshold=experiment_config.get("threshold", 0),
        loss_fn=end_to_end_model.loss,
        top_k=experiment_config.get("top_k", 32),
        temperature=experiment_config["temperature"],
        coherence_reg_weight=experiment_config["coherence_reg_weight"],
        diversity_reg_weight=experiment_config["diversity_reg_weight"],
        contrastive_reg_weight=experiment_config["contrastive_reg_weight"],
        feature_selection_reg_weight=experiment_config["feature_selection_reg_weight"],
        prob_diversity_reg_weight=experiment_config["prob_diversity_reg_weight"],
        concept_prediction_weight=experiment_config.get('concept_prediction_weight', 0),

        seed=experiment_config.get("seed", None),
        eps=experiment_config.get("eps", 1e-5),
        end_to_end_training=experiment_config.get('end_to_end_training', False),
        use_concept_embedding=experiment_config.get("use_concept_embedding", False),
        acc_metric=(
            (
                lambda y_true, y_pred: tf.math.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(
                    y_true,
                    y_pred,
                ))
            ) if experiment_config["num_outputs"] > 1 else tf.keras.metrics.binary_accuracy
        ),
        concept_generator_units=experiment_config.get('concept_generator_units', [64]),
        rec_model_units=experiment_config.get('rec_model_units', [64]),
        force_generator_inclusion=experiment_config.get('force_generator_inclusion', True),
    )
    tabcbm_model_path = os.path.join(
        experiment_config["results_dir"],
        f"models/tabcbm{extra_name}_weights/"
    )
    Path(tabcbm_model_path).mkdir(parents=True, exist_ok=True)
    if experiment_config.get('lr_schedule_decay', False):
        optimizer_gen = lambda: tf.keras.optimizers.Adam(
            tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=experiment_config.get(
                    'initial_learning_rate',
                    experiment_config.get("learning_rate", 1e-3),
                ),
                decay_steps=experiment_config.get('decay_steps', 10000),
                decay_rate=experiment_config.get('decay_rate', 0.9),
            )
        )
    else:
        optimizer_gen = lambda: tf.keras.optimizers.Adam(
            experiment_config.get("learning_rate", 1e-3),
        )
    
    if (
            (experiment_config.get("n_supervised_concepts", 0) != 0) and
            (len(experiment_config.get('supervised_concept_idxs', [])) > 0)
        ):
            # Then we will receive partial or full concept supervision here
            supervised_concept_idxs = experiment_config['supervised_concept_idxs']
            if 'selected_samples' in old_results:
                selected_samples = old_results['selected_samples']
            else:
                n_samples = c_train.shape[0]
                percent = experiment_config.get(
                    'concept_supervision_annotated_fraction',
                    1.0,
                )
                selected_samples = np.random.permutation(
                    list(range(n_samples))
                )[:int(np.ceil(n_samples * percent))]
                selected_samples = sorted(selected_samples)
                
            end_results['selected_samples'] = selected_samples
            c_train_real = np.empty((c_train.shape[0], len(supervised_concept_idxs)))
            c_train_real[:, :] = np.nan
            for i, idx in enumerate(supervised_concept_idxs):
                c_train_real[selected_samples, i] = c_train[selected_samples, idx]
            y_train_tensors = (y_train, c_train_real)
    else:
        y_train_tensors = y_train
        c_train_real = c_train
    if load_from_cache and os.path.exists(os.path.join(tabcbm_model_path, 'checkpoint')):
        logging.debug(prefix + "Found serialized TabCBM model! Unloading it right now...")
        tabcbm = TabCBM(
            self_supervised_mode=False,
            **tab_cbm_params,
        )
        tabcbm.compile(optimizer=optimizer_gen())
        tabcbm._compute_self_supervised_loss(x_test[:2, :])
        tabcbm._compute_supervised_loss(
            x_test[:2, :],
            y_test[:2],
            c_true=c_train_real[:2, :] if c_train_real is not None else None,
        )
        tabcbm(x_test[:2, :])
        tabcbm.load_weights(os.path.join(tabcbm_model_path, 'checkpoint'))
    
        ss_tabcbm_time_trained = old_results.get('ss_time_trained')
        ss_tabcbm_epochs_trained = old_results.get('ss_epochs_trained')
        tabcbm_time_trained = old_results.get('time_trained')
        tabcbm_epochs_trained = old_results.get('epochs_trained')
    
    else:
        
        # Else let's generate the model from scratch!
        ss_tabcbm = TabCBM(
            self_supervised_mode=True,
            **tab_cbm_params,
        )
        ss_tabcbm.compile(
            optimizer=tf.keras.optimizers.Adam(
                experiment_config.get("learning_rate", 1e-3),
            )
        )

        if experiment_config["self_supervised_train_epochs"]:
            logging.info(prefix + "TabCBM self-supervised training stage...")
            ss_tabcbm._compute_self_supervised_loss(x_test[:2, :])
            ss_tabcbm.set_weights(ss_tabcbm.get_weights())
            logging.debug(
                prefix +
                f"\t\tSelf-supervised model params: "
                f"{np.sum([np.prod(K.get_value(p).shape) for p in ss_tabcbm.trainable_weights])}"
            )
            early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
                monitor=experiment_config.get(
                    "early_stop_metric_ss",
                    "val_loss",
                ),
                min_delta=experiment_config["min_delta"],
                patience=experiment_config.get(
                    "patience_ss",
                    experiment_config["patience"],
                ),
                restore_best_weights=True,
                verbose=2,
                mode=experiment_config.get(
                    "early_stop_mode_ss",
                    "min",
                ),
            )
            if experiment_config.get('save_history', True):
                callbacks = [
                    early_stopping_monitor,
                    tf.keras.callbacks.CSVLogger(
                        os.path.join(
                            experiment_config["results_dir"],
                            "history",
                            (
                                f"ss_tabcbm{extra_name}_hist.csv"
                            )
                        ),
                        append=True
                    ),
                ]
            else:
                callbacks = [early_stopping_monitor]
            ss_tabcbm_hist, ss_tabcbm_time_trained = timeit(
                ss_tabcbm.fit,
                x=x_train,
                y=y_train_tensors,
                validation_split=experiment_config["holdout_fraction"],
                epochs=experiment_config["self_supervised_train_epochs"],
                batch_size=experiment_config["batch_size"],
                verbose=verbosity,
                callbacks=callbacks,
            )
            ss_tabcbm_epochs_trained = len(ss_tabcbm_hist.history['loss'])
            logging.debug(prefix + "\tTabCBM self-supervised training completed")
            end_results['ss_num_params'] = (
                np.sum([np.prod(K.get_value(p).shape) for p in ss_tabcbm.trainable_weights])
            )

            logging.info(prefix + "TabCBM supervised training stage...")
            
            if experiment_config.get('force_generator_inclusion', True):
                # Legacy mode where the model always includes the self-supervised stuff
                tabcbm = TabCBM(
                    self_supervised_mode=False,
                    **tab_cbm_params,
                )
                tabcbm.compile(optimizer=optimizer_gen())
                # Do a dummy call to initialize weights...
                tabcbm._compute_self_supervised_loss(x_test[:2, :])
                tabcbm._compute_supervised_loss(
                    x_test[:2, :],
                    y_test[:2],
                    c_true=c_train_real[:2, :] if c_train_real is not None else None,
                )
                tabcbm.set_weights(ss_tabcbm.get_weights())
            else:
                # else we only need to load the relevant stuff in here
                tabcbm = TabCBM(
                    self_supervised_mode=False,
                    concept_generators=ss_tabcbm.concept_generators,
                    prior_masks=ss_tabcbm.feature_probabilities,
                    **tab_cbm_params,
                )
                tabcbm.compile(optimizer=optimizer_gen())
                tabcbm._compute_supervised_loss(
                    x_test[:2, :],
                    y_test[:2],
                    c_true=c_train_real[:2, :] if c_train_real is not None else None,
                )
        else:
            ss_tabcbm_time_trained = 0
            ss_tabcbm_epochs_trained = 0
            tabcbm = ss_tabcbm
        early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
            monitor=experiment_config.get(
                "early_stop_metric",
                "val_loss",
            ),
            min_delta=experiment_config["min_delta"],
            patience=experiment_config.get(
                "patience",
                experiment_config["patience"],
            ),
            restore_best_weights=True,
            verbose=2,
            mode=experiment_config.get(
                "early_stop_mode",
                "min",
            ),
        )
        if experiment_config.get('save_history', True):
            callbacks = [
                early_stopping_monitor,
                tf.keras.callbacks.CSVLogger(
                    os.path.join(
                        experiment_config["results_dir"],
                        "history",
                        (
                            f"tabcbm{extra_name}_hist.csv"
                        )
                    ),
                    append=True
                ),
            ]
        else:
            callbacks = [early_stopping_monitor]
        
        tabcbm_hist, tabcbm_time_trained = timeit(
            tabcbm.fit,
            x=x_train,
            y=y_train_tensors,
            validation_split=experiment_config["holdout_fraction"],
            epochs=experiment_config["max_epochs"],
            batch_size=experiment_config["batch_size"],
            verbose=verbosity,
            callbacks=callbacks,
        )
        tabcbm_epochs_trained = len(tabcbm_hist.history['loss'])
        logging.debug(prefix + "\tTabCBM supervised training completed")

        logging.debug(prefix + "\tSerializing model")
        tabcbm.save_weights(os.path.join(tabcbm_model_path, 'checkpoint'))
    if 'ss_num_params' not in end_results and load_from_cache and (
        'ss_num_params' in old_results
    ):
        end_results['ss_num_params'] = old_results['ss_num_params']
    if 'ss_num_params' in end_results:
        logging.debug(prefix + f"\tNumber of self-supervised parameters = {end_results['ss_num_params']}")
    end_results['num_params'] = (
        np.sum([np.prod(p.shape) for p in tabcbm.trainable_weights])
    )
    logging.debug(prefix + f"\tNumber of TabCBM trainable parameters = {end_results['num_params']}")
    
    # Log training times and whatnot
    if pretrained_epochs_trained is not None:
        end_results['pretrained_epochs_trained'] = pretrained_epochs_trained
    if pretrained_time_trained is not None:
        end_results['pretrained_time_trained'] = pretrained_time_trained
    if tabcbm_epochs_trained is not None:
        end_results['epochs_trained'] = tabcbm_epochs_trained
    if tabcbm_time_trained is not None:
        end_results['time_trained'] = tabcbm_time_trained
    if ss_tabcbm_epochs_trained is not None:
        end_results['ss_epochs_trained'] = ss_tabcbm_epochs_trained
    if ss_tabcbm_time_trained is not None:
        end_results['ss_time_trained'] = ss_tabcbm_time_trained
    # Evaluate our model
    logging.info(prefix + "\tEvaluating TabCBM")
    test_output, test_concept_scores = tabcbm(x_test)
    test_concept_scores = test_concept_scores.numpy()
    test_output = test_output.numpy()
    if experiment_config["num_outputs"] > 1:
        # Then lets apply a softmax activation over all the probability
        # classes
        preds = scipy.special.softmax(
            test_output,
            axis=-1,
        )

        one_hot_labels = tf.keras.utils.to_categorical(y_test)
        end_results['acc'] = sklearn.metrics.accuracy_score(
            y_test,
            np.argmax(preds, axis=-1),
        )

        # And select just the labels that are in fact being used
        end_results['auc'] = sklearn.metrics.roc_auc_score(
            one_hot_labels,
            preds,
            multi_class='ovo',
        )
    else:
        end_results['acc'] = sklearn.metrics.accuracy_score(
            y_test,
            test_output,
        )
        end_results['auc'] = sklearn.metrics.roc_auc_score(
            y_test,
            test_output,
        )
    
    logging.debug(prefix + f"\t\tAccuracy is {end_results['acc']*100:.2f}%")
    
    if (
        (experiment_config.get("n_supervised_concepts", 0) != 0) and
        (len(experiment_config.get('supervised_concept_idxs', [])) > 0)
    ):
        # Then compute the mean concept predictive accuracy
        supervised_concept_idxs = experiment_config['supervised_concept_idxs']
        avg = 0.0
        for learnt_concept_idx, real_concept_idx in enumerate(
            supervised_concept_idxs
        ):
            # And select just the labels that are in fact being used
            avg += sklearn.metrics.roc_auc_score(
                c_test[:, real_concept_idx],
                test_concept_scores[:, learnt_concept_idx],
            )
        end_results['avg_concept_auc'] = avg / len(supervised_concept_idxs)
        logging.debug(
            prefix + f"\t\tMean Concept AUC is {end_results['avg_concept_auc']*100:.2f}%"
        )
    
    
    if (c_train is not None) and (c_test is not None):
        _, train_concept_scores = tabcbm(x_train)
        train_concept_scores = train_concept_scores.numpy()
        logging.debug(prefix + f"\t\tComputing best independent concept aligment...")
        end_results['best_independent_alignment'], end_results['best_ind_alignment_auc'] = posible_load(
            key=['best_independent_alignment', 'best_ind_alignment_auc'],
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.find_best_independent_alignment(
                scores=(train_concept_scores >= 0.5).astype(np.float32),
                c_train=c_train,
            ),
        )
        
        # Compute the CAS score
        logging.debug(prefix + "\t\tPredicting CAS...")
        end_results['cas'], end_results['cas_task'], end_results['best_alignment'] = posible_load(
            key=['cas', 'cas_task', 'best_alignment'],
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.embedding_homogeneity(
                c_vec=test_concept_scores,
                c_test=c_test,
                y_test=y_test,
                step=experiment_config.get('cas_step', 2),
            ),
        )
        logging.debug(prefix + f"\t\t\tDone with CAS = {end_results['cas'] * 100:.2f}%")
        
        if experiment_config.get('perform_interventions', True):
            # Then time to do some interventions!
            logging.debug(prefix + f"\t\tPerforming concept interventions")
            selected_concepts = end_results['best_ind_alignment_auc'] >= experiment_config.get(
                'usable_concept_threshold',
                0.85,
            )
            selected_concepts_idxs = np.array(list(range(experiment_config['n_concepts'])))[selected_concepts]
            corresponding_real_concepts = np.array(end_results['best_independent_alignment'])[selected_concepts]
            end_results['interveneable_concepts'] = posible_load(
                key='interveneable_concepts',
                old_results=old_results,
                load_from_cache=load_from_cache,
                run_fn=lambda: np.sum(selected_concepts),
            )
            logging.debug(
                prefix + f"\t\t\tNumber of concepts we will intervene on " +
                f"is {end_results['interveneable_concepts']}/{experiment_config['n_concepts']}"
            )
            _, test_bottleneck = tabcbm.predict_bottleneck(x_test)
            test_bottleneck = test_bottleneck.numpy()
            one_hot_labels = tf.keras.utils.to_categorical(y_test)
            for num_intervened_concepts in range(1, end_results['interveneable_concepts'] + 1):
                def _run():
                    avg = 0.0
                    for i in range(experiment_config.get('intervention_trials', 5)):
                        current_sel = np.random.permutation(
                            list(range(len(selected_concepts_idxs)))
                        )[:num_intervened_concepts]
                        fixed_used_concept_idxs = selected_concepts_idxs[current_sel]
                        real_corr_concept_idx = corresponding_real_concepts[current_sel]
                        new_test_bottleneck = test_bottleneck[:, :]
                        # We need to figure out the "direction" of the intervention:
                        #     There is not reason why a learnt concept aligned such that its
                        #     corresponding ground truth concept is high when the learnt concept
                        #     is high. Because they are binary, it could perfectly be the case
                        #     that the alignment happend with the complement.
                        for learnt_concept_idx, real_concept_idx in zip(
                            fixed_used_concept_idxs,
                            real_corr_concept_idx,
                        ):
                            correlation = np.corrcoef(
                                train_concept_scores[:, learnt_concept_idx],
                                c_train[:, real_concept_idx],
                            )[0, 1]
                            pos_score = np.percentile(train_concept_scores[:, learnt_concept_idx], 95)
                            neg_score = np.percentile(train_concept_scores[:, learnt_concept_idx], 5)
                            if correlation > 0:
                                # Then this is a positive alignment
                                new_test_bottleneck[:, learnt_concept_idx] = \
                                    c_test[:, real_concept_idx] * pos_score + (
                                        (1 - c_test[:, real_concept_idx]) * neg_score
                                    )
                            else:
                                # Else we are aligned with the complement
                                new_test_bottleneck[:, learnt_concept_idx] =  \
                                    (1 - c_test[:, real_concept_idx]) * pos_score + (
                                        c_test[:, real_concept_idx] * neg_score
                                    )
                        avg += sklearn.metrics.accuracy_score(
                            y_test,
                            np.argmax(
                                scipy.special.softmax(
                                    tabcbm.from_bottleneck(new_test_bottleneck),
                                    axis=-1,
                                ),
                                axis=-1
                            ),
                        )
                    return avg / experiment_config.get('intervention_trials', 5)

                end_results[f'acc_intervention_{num_intervened_concepts}'] =  posible_load(
                    key=f'acc_intervention_{num_intervened_concepts}',
                    old_results=old_results,
                    load_from_cache=load_from_cache,
                    run_fn=_run,
                )
                logging.debug(
                    prefix +
                    f"\t\t\tIntervention accuracy with {num_intervened_concepts} "
                    f"concepts: {end_results[f'acc_intervention_{num_intervened_concepts}'] * 100:.2f}%"
                )
        
    # Log statistics on the predicted masks
    masks = tf.sigmoid(tabcbm.feature_probabilities).numpy()
    end_results['mean_mask_prob'] = np.mean(masks)
    end_results['max_mask_prob'] = np.max(masks)
    end_results['min_mask_prob'] = np.min(masks)
    end_results['avg_num_of_features_per_concept'] = np.mean(np.sum(masks >= 0.5, axis=-1))
    
    if (ground_truth_concept_masks is not None) and (c_train is not None) and (
        c_test is not None
    ):
        # Then time to compute the best mask scores we can
        logging.debug(prefix + "\t\tPredicting best mean concept AUCs...")
        end_results['best_concept_auc'] = posible_load(
            key='best_concept_auc',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.brute_force_concept_aucs(
                concept_scores=test_concept_scores,
                c_test=c_test,
                reduction=np.mean,
                alignment=(
                    end_results['best_alignment']
                    if max(experiment_config['n_concepts'], c_test.shape[-1]) > 6
                    else None
                ),
            )['best_reduced_auc'],
        )
        logging.debug(prefix + f"\t\t\tDone: {end_results['best_concept_auc'] * 100:.2f}%")
        logging.debug(prefix + "\t\tPredicting best independent mean concept AUCs...")
        end_results['best_independent_concept_auc'] = posible_load(
            key='best_independent_concept_auc',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.brute_force_concept_aucs(
                concept_scores=test_concept_scores,
                c_test=c_test,
                reduction=np.mean,
                alignment=end_results['best_independent_alignment'],
            )['best_reduced_auc'],
        )
        logging.debug(prefix + f"\t\t\tDone: {end_results['best_independent_concept_auc'] * 100:.2f}%")
        logging.debug(prefix + "\t\tPredicting mean mask AUCs...")
        end_results['best_mean_mask_auc'] = posible_load(
            key='best_mean_mask_auc',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.brute_force_concept_mask_aucs(
                concept_importance_masks=masks,
                ground_truth_concept_masks=ground_truth_concept_masks,
                reduction=np.mean,
                alignment=(
                    end_results['best_alignment']
                    if max(experiment_config['n_concepts'], c_test.shape[-1]) > 6
                    else None
                ),
            )['best_reduced_auc'],
        )
        logging.debug(prefix + f"\t\t\tDone: {end_results['best_mean_mask_auc'] * 100:.2f}%")
        logging.debug(prefix + "\t\tPredicting mean independent mask AUCs...")
        end_results['best_independent_mean_mask_auc'] = posible_load(
            key='best_independent_mean_mask_auc',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.brute_force_concept_mask_aucs(
                concept_importance_masks=masks,
                ground_truth_concept_masks=ground_truth_concept_masks,
                reduction=np.mean,
                alignment=end_results['best_independent_alignment'],
            )['best_reduced_auc'],
        )
        logging.debug(prefix + f"\t\t\tDone: {end_results['best_independent_mean_mask_auc'] * 100:.2f}%")
        
        logging.debug(prefix + "\t\tPredicting max mask AUCs...")
        end_results['best_max_mask_auc'] = posible_load(
            key='best_max_mask_auc',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.brute_force_concept_mask_aucs(
                concept_importance_masks=masks,
                ground_truth_concept_masks=ground_truth_concept_masks,
                reduction=np.max,
                alignment=(
                    end_results['best_alignment']
                    if max(experiment_config['n_concepts'], c_test.shape[-1]) > 6
                    else None
                ),
            )['best_reduced_auc'],
        )
        logging.debug(prefix + f"\t\t\tDone: {end_results['best_max_mask_auc'] * 100:.2f}%")
        logging.debug(prefix + "\t\tPredicting max independent mask AUCs...")
        end_results['best_independent_max_mask_auc'] = posible_load(
            key='best_independent_max_mask_auc',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.brute_force_concept_mask_aucs(
                concept_importance_masks=masks,
                ground_truth_concept_masks=ground_truth_concept_masks,
                reduction=np.max,
                alignment=end_results['best_independent_alignment'],
            )['best_reduced_auc'],
        )
        
        logging.debug(prefix + f"\t\t\tDone: {end_results['best_independent_max_mask_auc'] * 100:.2f}%")
        logging.debug(prefix + "\t\tPredicting feature importance matching...")
        end_results['feat_importance_diff'] = posible_load(
            key='feat_importance_diff',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.feature_importance_diff(
                importance_masks=masks,
                c_train=c_train,
                ground_truth_concept_masks=ground_truth_concept_masks,
            ),
        )
        logging.debug(prefix + f"\t\t\tDone: {end_results['feat_importance_diff']:.5f}")
        logging.debug(prefix + "\t\tPredicting feature selection matching...")
        end_results['feat_selection'] = posible_load(
            key='feat_selection',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.feature_selection(
                importance_masks=masks,
                c_train=c_train,
                ground_truth_concept_masks=ground_truth_concept_masks,
            ),
        )
        logging.debug(prefix + f"\t\t\tDone: {end_results['feat_selection']:.5f}")
    if return_model:
        return end_results, tabcbm
    return end_results



############################################
## Main Experiment Loop
############################################

def initialize_result_directory(results_dir):
    Path(
        os.path.join(
            results_dir,
            "models",
        )
    ).mkdir(parents=True, exist_ok=True)
    
    Path(
        os.path.join(
            results_dir,
            "history",
        )
    ).mkdir(parents=True, exist_ok=True)
    
def experiment_loop(
    experiment_config,
    data_generator=None,
    x_train=None,
    y_train=None,
    c_train=None,
    x_test=None,
    y_test=None,
    c_test=None,
    load_from_cache=True,
    cov_mat=None,
    ground_truth_concept_masks=None,
    restart_gpu_on_run_trial=False,
    multiprocess_inference=True,
    print_cache_only=False,
    result_table_fields=None,
    sort_key="model",
):
    # Set log level in env variable as this will be necessary for
    # subprocessing
    os.environ['LOGLEVEL'] = os.environ.get(
        'LOGLEVEL',
        logging.getLevelName(logging.getLogger().getEffectiveLevel()),
    )
    loglevel = os.environ['LOGLEVEL']
    logging.info(f'Setting log level to: "{loglevel}"')
    # We will accumulate all results in a table for easier reading
    results_table = PrettyTable()
    field_names = [
        "Method",
        "Accuracy",
        "CAS",
        "Best Mean Concept Mask AUC",
        "Best Mean Concept AUC",
        "Feature Importance Diff",
        "Feature Selection AUC",
    ]
    result_table_fields_keys = [
        "acc",
        "cas",
        "best_mean_mask_auc",
        "best_concept_auc",
        "feat_importance_diff",
        "feat_selection",
    ]
    if result_table_fields is not None:
        for field in result_table_fields:
            if not isinstance(field, (tuple, list)):
                field = field, field
            field_name, field_pretty_name = field
            result_table_fields_keys.append(field_name)
            field_names.append(field_pretty_name)
    results_table.field_names = field_names

    # We will accumulate all results from all trials into this
    # dictionary so that we can later reduce them to compute their means/etc
    end_results = defaultdict(list)
    
    if data_generator is not None and x_train is not None:
        raise ValueError(
            'Either (x_train, x_test, y_train, y_test, c_train, c_test) is provided or '
            'a data generating function data_generator(seed) is provided, not both at '
            'the same time'
        )
    if data_generator is None and (
        (x_train is None) or
        (y_train is None) or
        (x_test is None) or
        (y_test is None)
    ):
        raise ValueError(
            'If data_generator is not provided, then we expect all '
            '(x_train, x_test, y_train, y_test) arguments to be '
            'explicitly provided'
        )
    
    # And time to iterate over all trials
    base_results_dir = experiment_config["results_dir"]
    partial_results = defaultdict(list)
    for trial in range(experiment_config["trials"]):
        # And then over all runs in a given trial
        extra_hypers = {}
        if data_generator is not None:
            # Then we generate fresh new data for each trial
            logging.info(f"Generating dataset for trial {trial + 1}/{experiment_config['trials']}...")
            data = data_generator(
                seed=trial,
                **experiment_config.get('data_hyperparams', {}),
            )
            if len(data) == 2 and isinstance(data[0], (tuple, list)) and isinstance(
                data[1],
                (dict,)
            ):
                data, extra_hypers = data
            if len(data) == 7:
                x_train, x_test, y_train, y_test, c_train, c_test, ground_truth_concept_masks = data
            elif len(data) == 6:
                x_train, x_test, y_train, y_test, c_train, c_test = data
                ground_truth_concept_masks = None
            elif len(data) == 4:
                x_train, x_test, y_train, y_test = data
                c_train = c_test = None
                ground_truth_concept_masks = None
            else:
                raise ValueError(
                    'Expected data generator to output tuple (x_train, x_test, y_train, y_test, c_train, c_test), '
                    'tuple (x_train, x_test, y_train, y_test), or tuple '
                    '(x_train, x_test, y_train, y_test, c_train, c_test, ground_truth_concept_masks). '
                    f'Instead we got a tuple with {len(data)} elements in it.'
                )
            logging.debug(f"\tDone!")
            
        logging.info(f"Train class distribution: {np.mean(tf.keras.utils.to_categorical(y_train), axis=0)}")
        logging.info(f"Test class distribution: {np.mean(tf.keras.utils.to_categorical(y_test), axis=0)}")
        logging.info(f"Train concept distribution: {np.mean(c_train, axis=0)}")
        logging.info(f"Test concept distribution: {np.mean(c_test, axis=0)}")
        for current_config in experiment_config['runs']:
            if restart_gpu_on_run_trial:
                device = cuda.get_current_device()
                device.reset()
            # Construct the config for this particular trial
            run_config = copy.deepcopy(experiment_config.get('shared_params', {}))
            run_config['input_shape'] = run_config.get(
                'input_shape',
                x_train.shape[1:],
            )
            logging.debug(f"\tInput shape is: {run_config['input_shape']}")
            if c_train is not None:
                run_config['n_ground_truth_concepts'] = run_config.get(
                    'n_ground_truth_concepts',
                    c_train.shape[-1],
                )
                logging.debug(f"\tNumber of ground truth concepts is: {run_config['n_ground_truth_concepts']}")
            run_config['num_outputs'] = run_config.get(
                'num_outputs',
                len(set(y_train)) if len(set(y_train)) > 2 else 1,
            )
            logging.debug(f"\tNumber of outputs is: {run_config['num_outputs']}")
                                              
            run_config.update(current_config)
            run_config.update(extra_hypers)
            _evaluate_expressions(run_config)
            arch = run_config['model']
            arch_name = arch.lower().strip()
            if arch_name == 'tabcbm':
                train_fn = train_tabcbm
                extra_kwargs = dict(
                    cov_mat=cov_mat,
                    ground_truth_concept_masks=ground_truth_concept_masks,
                )
            elif arch_name == "ccd":
                train_fn = train_ccd
                extra_kwargs = {}
            elif arch_name == "xgboost":
                train_fn = train_xgboost
                extra_kwargs = dict(
                    ground_truth_concept_masks=ground_truth_concept_masks,
                )
            elif arch_name == "lightgbm":
                train_fn = train_lightgbm
                extra_kwargs = dict(
                    ground_truth_concept_masks=ground_truth_concept_masks,
                )
            elif arch_name == "tabnet":
                train_fn = train_tabnet
                extra_kwargs = dict(
                    ground_truth_concept_masks=ground_truth_concept_masks,
                )
            elif arch_name == "mlp":
                train_fn = train_mlp
                extra_kwargs = {}
            elif arch_name == "senn":
                train_fn = train_senn
                extra_kwargs = {}
            else:
                raise ValueError(f'Unsupported model architecture "{arch}"')

            # Set up[ a local directory for this model to use for its results
            run_config["results_dir"] = os.path.join(base_results_dir, arch)
            initialize_result_directory(run_config["results_dir"])

            # Serialize the configuration we will be using for these experiments
            joblib.dump(
                run_config,
                os.path.join(run_config['results_dir'], "config.joblib"),
            )
            # Now time to actually train things and see what comes out
            # of this
            extra_name = run_config.get('extra_name', "").format(**run_config)
            if extra_name:
                extra_name = "_" + extra_name
            logging.info(
                f"\tRunning Trial {trial + 1}/{experiment_config['trials']} "
                f"for {arch}{extra_name}:"
            )
            nvidia_smi.nvmlInit()
            for i in range(nvidia_smi.nvmlDeviceGetCount()):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                logging.debug(
                    f"\tGPU device {i}: {nvidia_smi.nvmlDeviceGetName(handle)}, "
                    f"Memory : ({100*info.free/info.total:.2f}% free): "
                    f"{info.total}(total), {info.free} (free), {info.used} (used)"
                )
            nvidia_smi.nvmlShutdown()

            local_results_path = os.path.join(
                run_config["results_dir"],
                f"results{extra_name + f'_trial_{trial}'}.joblib",
            )
            old_results = None
            if os.path.exists(local_results_path):
                old_results = joblib.load(local_results_path)
            
            if (run_config.get('n_supervised_concepts', 0) > 0) and (c_train is not None):
                logging.debug(f"We will provide supervision for {run_config.get('n_supervised_concepts')} concepts")
                if load_from_cache and ('supervised_concept_idxs' in (old_results or {})):
                    supervised_concept_idxs = old_results['supervised_concept_idxs']
                else:
                    concept_idxs = np.random.permutation(list(range(run_config['n_ground_truth_concepts'])))
                    concept_idxs = sorted(concept_idxs[:run_config['n_supervised_concepts']])
                run_config['supervised_concept_idxs'] = concept_idxs
                logging.debug(f"\tSupervising on concepts {concept_idxs}")
            
            if print_cache_only and load_from_cache and (
                old_results is not None
            ):
                trial_results = old_results
            elif not multiprocess_inference:
                trial_results = train_fn(
                    experiment_config=run_config,
                    x_train=x_train,
                    y_train=y_train,
                    c_train=c_train,
                    x_test=x_test,
                    y_test=y_test,
                    c_test=c_test,
                    load_from_cache=load_from_cache,
                    prefix="\t\t\t",
                    seed=(trial + run_config.get('seed', 0)),
                    extra_name=(extra_name + f"_trial_{trial}"),
                    old_results=old_results,
                    return_model=False,
                    **extra_kwargs,
                )
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # We will run this as a subprocess as otherwise the memory allocated
                # for the GPU is not necessarily freed once the method returns
                manager = multiprocessing.Manager()
                trial_results = manager.dict()
                if arch_name in ["tabnet", "xgboost", "lightgbm"]:
                    context = multiprocessing
                else:
                    context = multiprocessing.get_context('spawn')
                p = context.Process(
                    target=train_fn,
                    kwargs=dict(
                        trial_results=trial_results,
                        experiment_config=run_config,
                        x_train=x_train,
                        y_train=y_train,
                        c_train=c_train,
                        x_test=x_test,
                        y_test=y_test,
                        c_test=c_test,
                        load_from_cache=load_from_cache,
                        prefix="\t\t\t",
                        seed=(trial + run_config.get('seed', 0)),
                        extra_name=(extra_name + f"_trial_{trial}"),
                        old_results=old_results,
                        return_model=False,
                        **extra_kwargs,
                    ),
                )
                p.start()
                logging.debug(f"\t\tStarting run in subprocess {p.pid}")
                p.join()
                if p.exitcode:
                    raise ValueError(
                        f'Subprocess for trial {trial + 1} of {arch}{extra_name} failed!'
                    )
                p.kill()

            logging.debug(
                f"\tTrial {trial + 1} COMPLETED for {arch}{extra_name}:"
            )
            # Figure out how we will aggregate results from different runs
            aggr_key = run_config.get('aggr_key', '{model}' + extra_name).format(
                **run_config
            )
            # And include them in our aggreation
            serialized_trial_results = {}  # Copy of trial results but it is a joblib-serializable copy
            for key, val in trial_results.items():
                if isinstance(val, list):
                    val = np.array(val)
                serialized_trial_results[key] = val
                partial_results[(aggr_key, key)].append(val)
                if isinstance(val, float) and int(val) != val:
                    # Then print it with some precision limit as well
                    # as displaying it in percent
                    logging.debug(f"\t\t{key} = {val:.4f}")
                else:
                    logging.debug(f"\t\t{key} = {val}")
            logging.debug(f"\t\tDone with trial {trial + 1}")
            # Locally serialize the results of this trial
            joblib.dump(serialized_trial_results, local_results_path)
    print("\t", "*" * 10, f"SUMMARY", "*" * 10)
    table_rows_inds = {name: i for (i, name) in enumerate(result_table_fields_keys)}
    table_rows = {}
    for (aggr_key, key), vals in partial_results.items():
        vals = np.array(vals)
        try:
            mean, std = np.mean(vals), np.std(vals)
            end_results[(aggr_key, key)].append((mean, std))
            if isinstance(vals[0], (float)) and int(vals[0]) != vals[0]:
                print(f"\t\t\t{aggr_key}__{key} = {mean:.4f} ± {std:.4f}")
            else:
                print(f"\t\t\t{aggr_key}__{key} = {mean} ± {std}")
            if aggr_key not in table_rows:
                table_rows[aggr_key] = [(None, None) for _ in result_table_fields_keys]
            if key in table_rows_inds:
                table_rows[aggr_key][table_rows_inds[key]] = (mean, std)
        except:
            # Else we could not average/reduce these results so we will save them as
            # they are.
            logging.warning(
                f"\tWe could not average results for {key} in model {aggr_key}"
            )
            end_results[(aggr_key, key)].append(vals)
    table_rows = list(table_rows.items())
    if sort_key == "model":
        # Then sort based on method name
        table_rows.sort(key=lambda x: x[0], reverse=True)
    elif sort_key in table_rows_inds:
        # Else sort based on the requested parameter
        table_rows.sort(
            key=lambda x: (
                x[1][table_rows_inds[sort_key]][0]
                if x[1][table_rows_inds[sort_key]][0] is not None else -float("inf")
            ),
            reverse=True,
    )
    for aggr_key, row in table_rows:
        for i, (mean, std) in enumerate(row):
            if mean is None or std is None:
                row[i] = "N/A"
            elif int(mean) == float(mean):
                row[i] = f'{mean} ± {std:}'
            else:
                row[i] = f'{mean:.4f} ± {std:.4f}'
        results_table.add_row([str(aggr_key)] + row)
    print("\t", "*" * 30)
    print(results_table)
    print("\n\n")

    # And serialize the results
    joblib.dump(
        end_results,
        os.path.join(
            base_results_dir,
            "results.joblib",
        ),
    )
    
    # Also serialize the results
    with open(os.path.join(base_results_dir, "output_table.txt"), "w") as f:
        f.write(str(results_table))
    return end_results

