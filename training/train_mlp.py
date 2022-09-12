import sklearn
import scipy
import tensorflow as tf
import numpy as np
from collections import defaultdict
import metrics
import os
import models.models as models
import logging
import training.utils as utils

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
    utils.restart_seeds(seed)
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
        end_to_end_hist, end_to_end_time_trained = utils.timeit(
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
