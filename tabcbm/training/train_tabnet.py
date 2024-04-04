import io
import joblib
import logging
import numpy as np
import os
import shutil
import sklearn

import tensorflow as tf
import torch
import zipfile

from pathlib import Path
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split

import tabcbm.metrics as metrics
import tabcbm.training.utils as utils

############################################
## Utils
############################################

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
    cat_feat_inds=[],
    cat_dims=[],
    ground_truth_concept_masks=None,
    trial_results=None,
    return_model=False,
):
    utils.restart_seeds(seed)
    end_results =  trial_results if trial_results is not None else {}
    old_results = (old_results or {}) if load_from_cache else {}
    verbosity = experiment_config.get("verbosity", 0)
    remap_cat_dims = []
    cat_dims = []
    for cat_dim in (cat_feat_inds or []):
        unique_vals = sorted(np.unique(x_train[:, cat_dim]).astype(np.int32))
        remap_cat_dims.append(dict([(val, i) for i, val in enumerate(unique_vals)]))
        cat_dims.append(len(unique_vals))
    if len(cat_feat_inds or []):
        for i in range(x_train.shape[0]):
            for remap, cat_dim in zip(remap_cat_dims, (cat_feat_inds or [])):
                x_train[i, cat_dim] = remap[x_train[i, cat_dim]]
        for i in range(x_test.shape[0]):
            for remap, cat_dim in zip(remap_cat_dims, (cat_feat_inds or [])):
                x_test[i, cat_dim] = remap[x_test[i, cat_dim]]
    tabnet_params = dict(
        n_d=experiment_config.get('n_d', 8),
        n_a=experiment_config.get('n_a', 8),
        n_steps=experiment_config.get('n_steps', 3),
        gamma=experiment_config.get('gamma', 1.3),
        cat_idxs=(cat_feat_inds or []),
        cat_dims=(cat_dims or []),
        cat_emb_dim=experiment_config.get("emb_out_size", 1),
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
        logging.debug(
            prefix + "Found TabNet model serialized! Loading it up..."
        )
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
            _, pretrain_time_trained = utils.timeit(
                pretrained_model.fit,
                X_train=x_train,
                eval_set=[x_val],
                max_epochs=experiment_config['pretrain_epochs'],
                pretraining_ratio=experiment_config.get('pretraining_ratio', 0.8),
                pin_memory=False, # Setting this to false to avoid hoarding the GPU
                **extra_params,
            )
            end_results['pretrain_num_parameters'] = sum([
                p.numel() for p in pretrained_model.network.parameters()
                if p.requires_grad
            ])

            if experiment_config.get("patience", float("inf")) not in [None, 0, float("inf")]:
                pretrain_epochs_trained = len(
                    pretrained_model._callback_container.callbacks[0].history["loss"]
                )
            else:
                pretrain_epochs_trained = experiment_config["pretrain_epochs"]

            extra_params["from_unsupervised"] = pretrained_model

            logging.debug(prefix + "\tDone!")
        else:
            pretrain_epochs_trained = None
            pretrain_time_trained = 0
        logging.info(prefix + "TabNet main model training...")

        tabnet = TabNetClassifier(**tabnet_params)
        _, time_trained = utils.timeit(
            tabnet.fit,
#             X_train=torch.cuda.FloatTensor(x_train),
            X_train=x_train,
#             y_train=torch.cuda.LongTensor(y_train),
            y_train=y_train,
            eval_set=eval_set,
            max_epochs=experiment_config["max_epochs"],
            pin_memory=False, # Setting this to false to avoid hoarding the GPU
            **extra_params,
        )
        end_results['num_parameters'] = sum([
            p.numel() for p in tabnet.network.parameters()
            if p.requires_grad
        ])
        if experiment_config.get("patience", float("inf")) not in [
            None,
            0,
            float("inf"),
        ]:
            epochs_trained = len(
                tabnet._callback_container.callbacks[0].history["loss"]
            )
        else:
            epochs_trained = experiment_config["max_epochs"]
        save_tabnet_model(tabnet, tabnet_path)
        logging.debug(prefix + "\tDone!")

    if 'pretrain_num_parameters' not in end_results and load_from_cache and (
        'pretrain_num_parameters' in old_results
    ):
        end_results['pretrain_num_parameters'] = old_results['pretrain_num_parameters']
    if 'pretrain_num_parameters' in end_results:
        logging.debug(
            prefix +
            f"\tNumber of pretrain parameters = {end_results['pretrain_num_parameters']}"
        )
    end_results['num_parameters'] = sum([
        p.numel() for p in tabnet.network.parameters()
        if p.requires_grad
    ])
    logging.debug(
        prefix +
        f"\tNumber of TabNet parameters = {end_results['num_parameters']}"
    )
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
        logging.debug(
            prefix + f"\t\t\tDone: {end_results['feat_importance_diff']:.5f}"
        )
        logging.debug(
            prefix + "\t\tPredicting feature selection matching..."
        )
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
