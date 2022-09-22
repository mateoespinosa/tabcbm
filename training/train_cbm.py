import sklearn
import scipy
import tensorflow as tf
import numpy as np
import metrics
import os
from pathlib import Path
import models.models as models
import joblib
import copy
from keras import backend as K
import logging
from concepts_xai.methods.CBM.CBModel import (
    JointConceptBottleneckModel,
    BypassJointCBM,
)

import training.utils as utils

############################################
## CBM Training
############################################

def train_cbm(
    experiment_config,
    x_train,
    y_train,
    c_train,
    x_test,
    y_test,
    c_test,
    load_from_cache=False,
    extra_name="",
    seed=0,
    old_results=None,
    prefix="",
    trial_results=None,
    return_model=False,
):
    utils.restart_seeds(seed)
    end_results = trial_results if trial_results is not None else {}
    old_results = (old_results or {}) if load_from_cache else {}
    verbosity = experiment_config.get("verbosity", 0)
    
    # Proceed to do and end-to-end model in case we want to
    # do some task-specific pretraining
    n_concepts = experiment_config['n_concepts']
    n_sup_concepts = experiment_config.get(
        "n_supervised_concepts",
        n_concepts,
    )
    extra_units = n_concepts - n_sup_concepts
    end_to_end_model, encoder, decoder = models.construct_end_to_end_model(
        input_shape=experiment_config["input_shape"],
        num_outputs=experiment_config["num_outputs"],
        learning_rate=experiment_config["learning_rate"],
        encoder=models.construct_encoder(
            input_shape=experiment_config["input_shape"],
            latent_dims=n_concepts,
            include_bn=experiment_config.get("include_bn", False),
            units=experiment_config["encoder_units"],
            latent_act=(
                None if experiment_config.get('pass_concept_logits', False)
                else tf.math.sigmoid
            ),
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
    if experiment_config.get('pretrain_epochs') and load_from_cache and (
        os.path.exists(encoder_path)
    ):
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
            pretrain_hist, pretrain_time_trained = utils.timeit(
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
    logging.debug(
        prefix + f"\t\tPretrained model task accuracy: {end_results['pre_train_acc']*100:.2f}%"
    )
    logging.debug(
        prefix +
        f"\t\tPretrained model params: "
        f"{np.sum([np.prod(K.get_value(p).shape) for p in end_to_end_model.trainable_weights])}"
    )
    
    cbm_model_path = os.path.join(
        experiment_config["results_dir"],
        f"models/cbm{extra_name}_weights/"
    )
    Path(cbm_model_path).mkdir(parents=True, exist_ok=True)
    
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
    
    # Now time to construct our CBM model
    cbm = JointConceptBottleneckModel(
        encoder=encoder,
        decoder=decoder,
        task_loss=(
            tf.keras.losses.BinaryCrossentropy(from_logits=True)
            if (experiment_config["num_outputs"] <= 2)
            else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        ),
        alpha=experiment_config['concept_loss_weight'],
        pass_concept_logits=experiment_config.get('pass_concept_logits', False),
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy")
            if (experiment_config['num_outputs'] <= 2) else
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        ]
    )
    cbm.compile(optimizer=optimizer_gen())
    
    if load_from_cache and os.path.exists(
        os.path.join(cbm_model_path, 'checkpoint')
    ):
        # Then we will load it from the cache
        logging.debug(
            prefix +
            "Found serialized CBM model! Unloading it right now..."
        )
        cbm(x_test[:2, :])
        cbm.load_weights(os.path.join(cbm_model_path, 'checkpoint'))
        cbm.compile(optimizer=optimizer_gen())
    
        cbm_time_trained = old_results.get('time_trained')
        cbm_epochs_trained = old_results.get('epochs_trained')
    else:
        # Else, time to train it from scratch
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
                            f"cbm{extra_name}_hist.csv"
                        )
                    ),
                    append=True
                ),
            ]
        else:
            callbacks = [early_stopping_monitor]
        
        cbm_hist, cbm_time_trained = utils.timeit(
            cbm.fit,
            x=x_train,
            y=y_train_tensors,
            validation_split=experiment_config["holdout_fraction"],
            epochs=experiment_config["max_epochs"],
            batch_size=experiment_config["batch_size"],
            verbose=verbosity,
            callbacks=callbacks,
        )
        cbm_epochs_trained = len(cbm_hist.history['loss'])
        logging.debug(
            prefix + "\tCBM supervised training completed"
        )

        logging.debug(prefix + "\tSerializing model")
        cbm.save_weights(os.path.join(cbm_model_path, 'checkpoint'))
    end_results['num_params'] = (
        np.sum([np.prod(p.shape) for p in cbm.trainable_weights])
    )
    logging.debug(
        prefix +
        f"\tNumber of CBM trainable parameters = {end_results['num_params']}"
    )
    
    # Log training times and whatnot
    if pretrained_epochs_trained is not None:
        end_results['pretrained_epochs_trained'] = pretrained_epochs_trained
    if pretrained_time_trained is not None:
        end_results['pretrained_time_trained'] = pretrained_time_trained
    if cbm_epochs_trained is not None:
        end_results['epochs_trained'] = cbm_epochs_trained
    if cbm_time_trained is not None:
        end_results['time_trained'] = cbm_time_trained

    # Evaluate our model
    logging.info(prefix + "\tEvaluating CBM")
    test_output, test_concept_scores = cbm(x_test)
    if isinstance(test_concept_scores, list):
        test_concept_scores = tf.concat(test_concept_scores, axis=-1)
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
    
    logging.debug(
        prefix + f"\t\tAccuracy is {end_results['acc']*100:.2f}%"
    )
    
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
            prefix +
            f"\t\tMean Concept AUC is {end_results['avg_concept_auc']*100:.2f}%"
        )
    
    
    if (c_train is not None) and (c_test is not None):
        _, train_concept_scores = cbm(x_train)
        if isinstance(train_concept_scores, list):
            train_concept_scores = tf.concat(train_concept_scores, axis=-1)
        train_concept_scores = train_concept_scores.numpy()
        logging.debug(prefix + f"\t\tComputing best independent concept aligment...")
        end_results['best_independent_alignment'], end_results['best_ind_alignment_auc'] = utils.posible_load(
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
        end_results['cas'], end_results['cas_task'], end_results['best_alignment'] = utils.posible_load(
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
        
        # Compute correlation between bottleneck entries and ground truch concepts
        logging.debug(prefix + "\t\tConcept correlation matrix...")
        end_results['concept_corr_mat'] = utils.posible_load(
            key='concept_corr_mat',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.correlation_alignment(
                scores=test_concept_scores,
                c_test=c_test,
            ),
        )
        logging.debug(prefix + f"\t\t\tDone")
        
        if experiment_config.get('perform_interventions', True):
            # Then time to do some interventions!
            logging.debug(prefix + f"\t\tPerforming concept interventions")
            selected_concepts = end_results['best_ind_alignment_auc'] >= experiment_config.get(
                'usable_concept_threshold',
                0.85,
            )
            selected_concepts_idxs = np.array(
                list(range(experiment_config['n_concepts']))
            )[selected_concepts]
            corresponding_real_concepts = np.array(
                end_results['best_independent_alignment']
            )[selected_concepts]
            end_results['interveneable_concepts'] = utils.posible_load(
                key='interveneable_concepts',
                old_results=old_results,
                load_from_cache=load_from_cache,
                run_fn=lambda: np.sum(selected_concepts),
            )
            logging.debug(
                prefix + f"\t\t\tNumber of concepts we will intervene on " +
                f"is {end_results['interveneable_concepts']}/{experiment_config['n_concepts']}"
            )
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
                        new_test_bottleneck = test_concept_scores[:, :]
                        # We need to figure out the "direction" of the intervention:
                        #     There is not reason why a learnt concept aligned such that its
                        #     corresponding ground truth concept is high when the learnt concept
                        #     is high. Because they are binary, it could perfectly be the case
                        #     that the alignment happend with the complement.
                        for learnt_concept_idx, real_concept_idx in zip(
                            fixed_used_concept_idxs,
                            real_corr_concept_idx,
                        ):
                            pos_score = 1
                            neg_score = 0
                            new_test_bottleneck[:, learnt_concept_idx] = \
                                c_test[:, real_concept_idx] * pos_score + (
                                    (1 - c_test[:, real_concept_idx]) * neg_score
                                )
                        avg += sklearn.metrics.accuracy_score(
                            y_test,
                            np.argmax(
                                scipy.special.softmax(
                                    cbm.predict_from_concepts(new_test_bottleneck),
                                    axis=-1,
                                ),
                                axis=-1
                            ),
                        )
                    return avg / experiment_config.get('intervention_trials', 5)

                end_results[f'acc_intervention_{num_intervened_concepts}'] = utils.posible_load(
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
        
    if return_model:
        return end_results, cbm
    return end_results
