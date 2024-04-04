import logging
import numpy as np
import os
import scipy
import sklearn
import tensorflow as tf

import tabcbm.models.models as models
import tabcbm.training.representation_evaluation as representation_evaluation
import tabcbm.training.utils as utils
import tabcbm.concepts_xai.methods.VAE.betaVAE as beta_vae

############################################
## Utils
############################################

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
    cat_feat_inds=None,
    cat_dims=None,
):
    utils.restart_seeds(seed)
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
        emb_dims=cat_feat_inds,
        emb_in_size=cat_dims,
        emb_out_size=experiment_config.get("emb_out_size", 1),
    )
    concept_decoder = models.construct_vae_decoder(
        units=experiment_config["decoder_units"],
        output_shape=experiment_config["input_shape"][-1],
        latent_dims=experiment_config['n_concepts'],
    )
    coefficient_model = models.construct_senn_coefficient_model(
        units=experiment_config["coefficient_model_units"],
        num_concepts=experiment_config['n_concepts'],
        num_outputs=max(experiment_config["num_outputs"], 2),
    )
    encoder_path = os.path.join(
        experiment_config["results_dir"],
        f"models/encoder{extra_name}"
    )
    if load_from_cache and os.path.exists(encoder_path):
        logging.debug(prefix + "Found cached SENN model! Loading it...")
        concept_encoder = tf.keras.models.load_model(encoder_path)
        concept_decoder = tf.keras.models.load_model(
            encoder_path.replace('/encoder_', '/decoder_')
        )
        coefficient_model = tf.keras.models.load_model(
            encoder_path.replace('/encoder_', '/coefficient_model_')
        )
        senn_model = models.construct_senn_model(
            concept_encoder=concept_encoder,
            concept_decoder=concept_decoder,
            coefficient_model=coefficient_model,
            num_outputs=max(experiment_config["num_outputs"], 2),
            regularization_strength=experiment_config.get(
                "regularization_strength",
                0.1,
            ),
            learning_rate=experiment_config.get("learning_rate", 1e-3),
            sparsity_strength=experiment_config.get(
                "sparsity_strength",
                2e-5,
            ),
        )
        senn_epochs_trained = old_results.get('epochs_trained')
        senn_time_trained = old_results.get('time_trained')
        autoencoder_epochs_trained = old_results.get('autoencoder_epochs_trained')
        autoencoder_time_trained = old_results.get('autoencoder_time_trained')
    else:
        if experiment_config.get("pretrain_autoencoder_epochs"):
            logging.info(prefix + "SENN Autoencoder model pre-training...")
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
            autoencoder_hist, autoencoder_time_trained = utils.timeit(
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
            num_outputs=max(experiment_config["num_outputs"], 2),
            regularization_strength=experiment_config.get(
                "senn_regularization_strength",
                0.1,
            ),
            learning_rate=experiment_config.get("learning_rate", 1e-3),
            sparsity_strength=experiment_config.get(
                "senn_sparsity_strength",
                2e-5,
            ),
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

        senn_hist, senn_time_trained = utils.timeit(
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
    test_output, (_, x_test_theta_class_scores) = senn_model.predict(
        x_test,
        batch_size=experiment_config["batch_size"],
    )
    test_concept_scores = get_argmax_concept_explanations(
        test_output,
        x_test_theta_class_scores,
    )

    # Compute the model's accuracies
    logging.debug(prefix + "\t\tComputing accuracies...")
    if max(experiment_config["num_outputs"], 2) > 1:
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

    logging.debug(prefix + "\t\tDone with evaluation...")
    if return_model:
        return end_results, senn_model
    return end_results
