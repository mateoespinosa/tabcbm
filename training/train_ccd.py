import sklearn
import scipy
import tensorflow as tf
import numpy as np
import metrics
import concepts_xai.methods.OCACE.topicModel as CCD
import concepts_xai.evaluation.metrics.completeness as completeness
import os
import models.models as models
from keras import backend as K
import logging

import training.utils as utils
import training.representation_evaluation as representation_evaluation


############################################
## CCD Training
############################################

def ccd_compute_k(y, batch_size):
    _, counts = np.unique(y, return_counts=True)
    avg_class_ratio = np.mean(counts) / y.shape[0]
    return int((avg_class_ratio * batch_size) / 2)

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
    cat_feat_inds=None,
    cat_dims=None,
):
    utils.restart_seeds(seed)
    channels_axis = (
        -1 if experiment_config.get("data_format", "channels_last") == "channels_last"
        else 1
    )

    end_results = trial_results if trial_results is not None else {}
    old_results = (old_results or {}) if load_from_cache else {}
    verbosity = experiment_config.get("verbosity", 0)

    if experiment_config["num_outputs"] < 2:
        acc_fn = lambda y_true, y_pred: sklearn.metrics.roc_auc_score(
            y_true,
            y_pred
        )
    else:
        acc_fn = lambda y_true, y_pred: sklearn.metrics.accuracy_score(
            y_true,
            np.argmax(y_pred, axis=-1),
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
            emb_dims=cat_feat_inds,
            emb_in_size=cat_dims,
            emb_out_size=experiment_config.get("emb_out_size", 1),
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
        logging.debug(
            prefix +
            "Found encoder/decoder models serialized! We will unload them " +
            "into the end-to-end model!"
        )
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
        end_to_end_hist, end_to_end_time_trained = utils.timeit(
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
    end_to_end_preds = end_to_end_model.predict(
        x_test,
        batch_size=experiment_config["batch_size"],
    )
    if experiment_config["num_outputs"] > 1:
        preds = end_to_end_preds
        if np.min(preds) < 0.0 or np.max(preds) > 1:
            preds = scipy.special.softmax(
                preds,
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
        if np.min(end_to_end_preds) < 0.0 or np.max(end_to_end_preds) > 1:
            # Then we assume that we have outputed logits
            end_to_end_preds = tf.math.sigmoid(end_to_end_preds).numpy()
        end_results['pre_train_acc'] = sklearn.metrics.accuracy_score(
            y_test,
            (end_to_end_preds >= 0.5).astype(np.int32),
        )
        end_results['pre_train_auc'] = sklearn.metrics.roc_auc_score(
            y_test,
            (end_to_end_preds >= 0.5).astype(np.int32),
        )
    logging.debug(
        prefix + f"\t\tPretrained model task accuracy: "
        f"{end_results['pre_train_acc']*100:.2f}%"
    )

    # Now extract our concept vectors
    def derp(y_true, y_pred):
        return tf.keras.metrics.binary_accuracy(
            tf.cast(y_true, tf.int32),
            tf.cast(tf.math.sigmoid(y_pred) >= 0.5, tf.int32)
        )
    if experiment_config.get('top_k', None) not in [None, 0]:
        top_k = experiment_config['top_k']
    else:
        # Else let's compute top k as suggested by Yeh et al.
        top_k = ccd_compute_k(y=y_train, batch_size=experiment_config["batch_size"])
    print("CCD's top k is:", top_k)
    topic_model = CCD.TopicModel(
        concepts_to_labels_model=decoder,
        n_channels=experiment_config["latent_dims"],
        n_concepts=experiment_config['n_concepts'],
        threshold=experiment_config.get("threshold", 0.5),
        loss_fn=end_to_end_model.loss,
        top_k=top_k,
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
            ) if experiment_config["num_outputs"] > 1 else derp #lambda y_true, y_pred: tf.keras.metrics.binary_accuracy(
#                 y_true,
#                 tf.cast(y_pred >= 0.5, tf.int32),
#             )
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
        train_encodings = encoder.predict(
            x_train,
            batch_size=experiment_config["batch_size"],
        )
        ccd_hist, ccd_time_trained = utils.timeit(
            topic_model.fit,
            x=train_encodings,
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
    test_encodings = encoder.predict(
        x_test,
        batch_size=experiment_config["batch_size"],
    )
    test_output, test_concept_scores = topic_model.predict(
        test_encodings,
        batch_size=experiment_config["batch_size"],
    )
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
        preds = test_output
        if np.min(preds) < 0.0 or np.max(preds) > 1:
            # Then we assume that we have outputed logits
            preds = tf.math.sigmoid(preds).numpy()
        end_results['acc'] = sklearn.metrics.accuracy_score(
            y_test,
            (preds >= 0.5).astype(np.int32),
        )
        end_results['auc'] = sklearn.metrics.roc_auc_score(
            y_test,
            (preds >= 0.5).astype(np.int32),
        )

    # Compute the CAS score
    if (not experiment_config.get('continuous_concepts', False)) and (
        c_test is not None
    ):
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

    # Let's see our topic model's completeness
    logging.debug(prefix + f"\t\tComputing CCD's completeness scores...")
    end_results['completeness']= utils.posible_load(
        key='completeness',
        old_results=old_results,
        load_from_cache=load_from_cache,
        run_fn=lambda: completeness.completeness_score(
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
        )[0],
    )

    end_results['direct_completeness'] = utils.posible_load(
        key='direct_completeness',
        old_results=old_results,
        load_from_cache=load_from_cache,
        run_fn=lambda: completeness.direct_completeness_score(
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
        )[0],
    )

    if return_model:
        return end_results, topic_model
    return end_results

