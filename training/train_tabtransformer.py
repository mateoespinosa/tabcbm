import sklearn
import tensorflow as tf
import numpy as np
import metrics
import os
from pathlib import Path
import joblib
from models.tabtransformer import TabTransformer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
import torch
import shutil
import zipfile
import copy
import logging

import training.utils as utils


############################################
## TabTransformer Training
############################################

def train_tabtransformer(
    experiment_config,
    x_train,
    y_train,
    c_train,
    x_test,
    y_test,
    c_test,
    cat_feat_inds=[],
    cat_dims=[],
    load_from_cache=False,
    seed=0,
    old_results=None,
    prefix="",
    extra_name='',
    ground_truth_concept_masks=None,
    trial_results=None,
    return_model=False,
):
    utils.restart_seeds(seed)
    end_results =  trial_results if trial_results is not None else {}
    cat_feat_inds = cat_feat_inds or []
    cat_dims = cat_dims or []
    old_results = (old_results or {}) if load_from_cache else {}
    verbosity = experiment_config.get("verbosity", 0)
    
    num_continuous = x_train.shape[1] - len(cat_feat_inds)
    cont_idxs = [i for i in range(x_train.shape[1]) if i not in cat_feat_inds]
    tabtransformer_params = dict(
        cat_idxs=cat_feat_inds,
        cont_idxs=cont_idxs,
        categories=cat_dims,
        num_continuous=num_continuous,
        dim=experiment_config.get('dim', 32),
        dim_out=len(np.unique(y_train)),
        depth=experiment_config.get('depth', 6),
        heads=experiment_config.get('heads', 8),
        attn_dropout=experiment_config.get('attn_dropout', 0.1),
        ff_dropout=experiment_config.get('ff_dropout', 0.1),
        mlp_hidden_mults=experiment_config.get('mlp_hidden_mults', (4, 2)),
        mlp_act=torch.nn.ReLU(),
        learning_rate=experiment_config['learning_rate'],
        momentum=experiment_config.get('momentum', 0.9),
        weight_decay=experiment_config.get('weight_decay', 0),
        optimizer=experiment_config.get('optimizer', 'adam'),
    )
    
    tabtransformer_path = os.path.join(
        experiment_config["results_dir"],
        f"models"
    )
    Path(tabtransformer_path).mkdir(parents=True, exist_ok=True)
    tabtransformer_path = os.path.join(tabtransformer_path, f'tabtransformer{extra_name}.pt')
    
    callbacks = []
    if experiment_config.get("patience", float("inf")) not in [
        None,
        0,
        float("inf"),
    ]:
        early_stop = EarlyStopping(
            monitor=experiment_config.get(
                "early_stop_metric",
                "val_loss",
            ),
            mode=experiment_config.get(
                "early_stop_mode",
                "min",
            ),
            patience=experiment_config["patience"],
        )
        callbacks = [early_stop]
        
    trainer = pl.Trainer(
        gpus=int(torch.cuda.is_available()),
        max_epochs=experiment_config['max_epochs'],
        check_val_every_n_epoch=experiment_config.get("check_val_every_n_epoch", 5),
        callbacks=callbacks,
        logger=True,
    )
    
    if experiment_config["holdout_fraction"]:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=experiment_config["holdout_fraction"],
            random_state=42,
        )
        val_data = torch.utils.data.TensorDataset(
            torch.cuda.FloatTensor(x_val),
            torch.cuda.FloatTensor(y_val),
        )
        val_dl = torch.utils.data.DataLoader(
            val_data,
            batch_size=experiment_config["batch_size"],
        )
    else:
        val_dl = None
    train_data = torch.utils.data.TensorDataset(
        torch.cuda.FloatTensor(x_train),
        torch.cuda.FloatTensor(y_train),
    )
    train_dl = torch.utils.data.DataLoader(
        train_data,
        batch_size=experiment_config["batch_size"],
    )
    
    if  load_from_cache and os.path.exists(tabtransformer_path):
        logging.debug(
            prefix + "Found TabTranasformer model serialized! Loading it up..."
        )
        # Then time to load up the end-to-end model!
        tabtransformer = TabTransformer(**tabtransformer_params)
        tabtransformer.load_state_dict(torch.load(tabtransformer_path))
        epochs_trained = old_results.get('epochs_trained')
        time_trained = old_results.get('time_trained')
    else:
        logging.info(prefix + "TabTransformer training...")
        tabtransformer = TabTransformer(**tabtransformer_params)
        _, time_trained = utils.timeit(
            trainer.fit,
            model=tabtransformer,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
        )
        if experiment_config.get("patience", float("inf")) not in [
            None,
            0,
            float("inf"),
        ]:
            epochs_trained = early_stop.stopped_epoch
        else:
            epochs_trained = experiment_config["max_epochs"]
        torch.save(
            tabtransformer.state_dict(),
            tabtransformer_path,
        )
        logging.debug(prefix + "\tDone!")
    
    end_results['num_params'] = sum([
        p.numel() for p in tabtransformer.parameters()
        if p.requires_grad
    ])
    logging.debug(
        prefix +
        f"\tNumber of TabTransformer parameters = {end_results['num_params']}"
    )
    logging.info(prefix + "\tEvaluating TabTransformer model")
    test_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.cuda.FloatTensor(x_test),
            torch.cuda.FloatTensor(y_test),
        ),
        batch_size=experiment_config["batch_size"],
    )
    preds = trainer.predict(tabtransformer, test_dl)
    preds = np.concatenate(
        preds,
        axis=0,
    )
    
    if experiment_config["num_outputs"] > 1:
        one_hot_labels = tf.keras.utils.to_categorical(y_test)
        preds = np.argmax(
            preds,
            axis=-1,
        )
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
    
    # Log training times and whatnot
    if epochs_trained is not None:
        end_results['epochs_trained'] = epochs_trained
    if time_trained is not None:
        end_results['time_trained'] = time_trained
    if return_model:
        return end_results, tabtransformer
    del tabtransformer
    return end_results
