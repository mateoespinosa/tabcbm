import sklearn
import scipy
import tensorflow as tf
import numpy as np
import metrics
import os
from pathlib import Path
import models.models as models
from models.tabcbm import TabCBM
from keras import backend as K
import logging

import training.utils as utils

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
    utils.restart_seeds(seed)
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
    end_to_end_preds = end_to_end_model.predict(
        x_test,
        batch_size=experiment_config["batch_size"],
    )
    if ((len(end_to_end_preds.shape) == 2)) and (end_to_end_preds.shape[-1] >= 2):
        preds = scipy.special.softmax(
            end_to_end_preds,
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
        if np.min(end_to_end_preds) < 0.0 or np.max(end_to_end_pred) > 1:
            # Then we assume that we have outputed logits
            end_to_end_preds = tf.math.sigmoid(end_to_end_preds).numpy()
        end_to_end_preds = (end_to_end_preds >= 0.5).astype(np.int32)
        end_results['pre_train_acc'] = sklearn.metrics.accuracy_score(
            y_test,
            end_to_end_preds,
        )
        end_results['pre_train_auc'] = sklearn.metrics.roc_auc_score(
            y_test,
            end_to_end_preds,
        )
    logging.debug(prefix + f"\t\tPretrained model task accuracy: {end_results['pre_train_acc']*100:.2f}%")
    logging.debug(
        prefix +
        f"\t\tPretrained model params: "
        f"{np.sum([np.prod(K.get_value(p).shape) for p in end_to_end_model.trainable_weights])}"
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
        feature_budget=experiment_config.get('feature_budget'),
        feature_budget_weight=experiment_config.get('feature_budget_weight', 0),
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
        logging.debug(
            prefix +
            "Found serialized TabCBM model! Unloading it right now..."
        )
        tabcbm = TabCBM(
            self_supervised_mode=False,
            **tab_cbm_params,
        )
        tabcbm.compile(optimizer=optimizer_gen())
        if experiment_config.get('force_generator_inclusion', True):
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
            ss_tabcbm_hist, ss_tabcbm_time_trained = utils.timeit(
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
                np.sum([
                    np.prod(K.get_value(p).shape)
                    for p in ss_tabcbm.trainable_weights
                ])
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
                    c_true=(
                        c_train_real[:2, :]
                        if c_train_real is not None else None
                    ),
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
        logging.debug(
            prefix +
            f"\tTabCBM trainable parameters = " + 
            f"{np.sum([np.prod(p.shape) for p in tabcbm.trainable_weights])}"
        )
        
        tabcbm_hist, tabcbm_time_trained = utils.timeit(
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
        logging.debug(
            prefix + "\tTabCBM supervised training completed"
        )

        logging.debug(prefix + "\tSerializing model")
        tabcbm.save_weights(os.path.join(tabcbm_model_path, 'checkpoint'))
    if 'ss_num_params' not in end_results and load_from_cache and (
        'ss_num_params' in old_results
    ):
        end_results['ss_num_params'] = old_results['ss_num_params']
    if 'ss_num_params' in end_results:
        logging.debug(
            prefix +
            f"\tNumber of self-supervised parameters = {end_results['ss_num_params']}"
        )
    end_results['num_params'] = (
        np.sum([np.prod(p.shape) for p in tabcbm.trainable_weights])
    )
    logging.debug(
        prefix +
        f"\tNumber of TabCBM trainable parameters = {end_results['num_params']}"
    )
    
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
    test_output, test_concept_scores = tabcbm.predict(
        x_test,
        batch_size=experiment_config["batch_size"],
    )
    if ((len(test_output.shape) == 2)) and (test_output.shape[-1] >= 2):
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
        test_preds = test_output
        if np.min(test_preds) < 0.0 or np.max(test_preds) > 1:
            # Then we assume that we have outputed logits
            test_preds = tf.math.sigmoid(test_preds).numpy()
        test_preds = (test_preds >= 0.5).astype(np.int32)
        end_results['acc'] = sklearn.metrics.accuracy_score(
            y_test,
            test_preds,
        )
        end_results['auc'] = sklearn.metrics.roc_auc_score(
            y_test,
            test_preds,
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
        _, train_concept_scores = tabcbm.predict(
            x_train,
            batch_size=experiment_config["batch_size"],
        )
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
        
        if not experiment_config.get('continuous_concepts', False):
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
        if experiment_config.get('continuous_concepts', False):
            print("Masks:")
            print(tf.sigmoid(tabcbm.feature_probabilities).numpy())
            print("Concept correlation matrix:")
            print(end_results['concept_corr_mat'])
        logging.debug(prefix + f"\t\t\tDone")
        
        if experiment_config.get('perform_interventions', True) and (
            not experiment_config.get('continuous_concepts', False)
        ):
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
                            pos_score = np.percentile(
                                train_concept_scores[:, learnt_concept_idx],
                                95
                            )
                            neg_score = np.percentile(
                                train_concept_scores[:, learnt_concept_idx],
                                5
                            )
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
        
    # Log statistics on the predicted masks
    masks = tf.sigmoid(tabcbm.feature_probabilities).numpy()
    end_results['mean_mask_prob'] = np.mean(masks)
    end_results['max_mask_prob'] = np.max(masks)
    end_results['min_mask_prob'] = np.min(masks)
    end_results['avg_num_of_features_per_concept'] = np.mean(
        np.sum(masks >= 0.5, axis=-1)
    )
    
    if (ground_truth_concept_masks is not None) and (c_train is not None) and (
        c_test is not None
    ):
        # Then time to compute the best mask scores we can
        if not experiment_config.get('continuous_concepts', False):
            logging.debug(prefix + "\t\tPredicting best mean concept AUCs...")
            end_results['best_concept_auc'] = utils.posible_load(
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
            logging.debug(
                prefix +
                f"\t\t\tDone: {end_results['best_concept_auc'] * 100:.2f}%"
            )
            logging.debug(
                prefix +
                "\t\tPredicting best independent mean concept AUCs..."
            )
            end_results['best_independent_concept_auc'] = utils.posible_load(
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
            logging.debug(
                prefix +
                f"\t\t\tDone: {end_results['best_independent_concept_auc'] * 100:.2f}%"
            )
        logging.debug(prefix + "\t\tPredicting mean mask AUCs...")
        end_results['best_mean_mask_auc'] = utils.posible_load(
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
        logging.debug(
            prefix +
            f"\t\t\tDone: {end_results['best_mean_mask_auc'] * 100:.2f}%"
        )
        logging.debug(
            prefix +
            "\t\tPredicting mean independent mask AUCs..."
        )
        end_results['best_independent_mean_mask_auc'] = utils.posible_load(
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
        logging.debug(
            prefix +
            f"\t\t\tDone: {end_results['best_independent_mean_mask_auc'] * 100:.2f}%"
        )
        
        logging.debug(prefix + "\t\tPredicting max mask AUCs...")
        end_results['best_max_mask_auc'] = utils.posible_load(
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
        logging.debug(
            prefix +
            f"\t\t\tDone: {end_results['best_max_mask_auc'] * 100:.2f}%"
        )
        logging.debug(prefix + "\t\tPredicting max independent mask AUCs...")
        end_results['best_independent_max_mask_auc'] = utils.posible_load(
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
        
        logging.debug(
            prefix +
            f"\t\t\tDone: {end_results['best_independent_max_mask_auc'] * 100:.2f}%"
        )
        logging.debug(prefix + "\t\tPredicting feature importance matching...")
        end_results['feat_importance_diff'] = utils.posible_load(
            key='feat_importance_diff',
            old_results=old_results,
            load_from_cache=load_from_cache,
            run_fn=lambda: metrics.feature_importance_diff(
                importance_masks=masks,
                c_train=c_train,
                ground_truth_concept_masks=ground_truth_concept_masks,
            ),
        )
        logging.debug(
            prefix + f"\t\t\tDone: {end_results['feat_importance_diff']:.5f}"
        )
        logging.debug(prefix + "\t\tPredicting feature selection matching...")
        end_results['feat_selection'] = utils.posible_load(
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
