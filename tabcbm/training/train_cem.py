import logging
import numpy as np
import os
import pytorch_lightning as pl
import scipy
import sklearn
import tensorflow as tf
import torch


from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split

import tabcbm.metrics as metrics
import tabcbm.models.models as models
import tabcbm.training.representation_evaluation as representation_evaluation
import tabcbm.training.utils as utils
from tabcbm.models.cem import ConceptEmbeddingModel

class MLP(pl.LightningModule):
    def __init__(self, input_shape, units, include_bn=False):
        super().__init__()
        curr_size = np.prod(input_shape)
        all_layers = [torch.nn.Flatten()]
        if include_bn:
            all_layers.append(
                torch.nn.BatchNorm1d(num_features=curr_size)
            )
        for acts in units[:-1]:
            all_layers.extend([torch.nn.Linear(curr_size, acts), torch.nn.ReLU()])
            if include_bn:
                all_layers.append(
                    torch.nn.BatchNorm1d(num_features=acts)
                )
            curr_size = acts
        all_layers.extend([torch.nn.Linear(curr_size, units[-1]), torch.nn.ReLU()])
        all_layers[-1].output_features = units[-1]
        self.model = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        return self.model(x)


############################################
## CEM Training
############################################

def train_cem(
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

    def c_extractor_arch(output_dim):
        return MLP(
            units=experiment_config["encoder_units"] + [output_dim],
            input_shape=experiment_config["input_shape"],
            include_bn=experiment_config.get("include_bn", False),
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
    else:
        c_train_real = c_train

    # Now time to construct our CEM model
    cem_params = dict(
        n_concepts=n_concepts,
        n_tasks=len(np.unique(y_train)),
        concept_loss_weight=experiment_config['concept_loss_weight'],
        task_loss_weight=experiment_config.get('task_loss_weight', 1.0),
        learning_rate=experiment_config['learning_rate'],
        weight_decay=experiment_config.get('weight_decay', 0),
        c_extractor_arch=c_extractor_arch,
        emb_size=experiment_config.get("emb_size", 8),
        n_latent_acts=experiment_config.get("n_latent_acts", experiment_config["encoder_units"][-1]),
        top_k_accuracy=None,
    )
    cem = ConceptEmbeddingModel(**cem_params)

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

    cem_model_path = os.path.join(
        experiment_config["results_dir"],
        f"models"
    )
    Path(cem_model_path).mkdir(parents=True, exist_ok=True)
    cem_model_path = os.path.join(cem_model_path, f'cem{extra_name}_end.pt')

    if experiment_config["holdout_fraction"]:
        if (c_train_real is not None) and (c_test is not None):
            x_train, x_val, y_train, y_val, c_train_real, c_val_real, c_train, c_val = train_test_split(
                x_train,
                y_train,
                c_train_real,
                c_train,
                test_size=experiment_config["holdout_fraction"],
                random_state=42,
            )
            val_data = torch.utils.data.TensorDataset(
                torch.cuda.FloatTensor(x_val),
                torch.cuda.FloatTensor(y_val),
                torch.cuda.FloatTensor(c_val_real),
            )
        else:
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
    if (c_train_real is not None):
        train_data = torch.utils.data.TensorDataset(
            torch.cuda.FloatTensor(x_train),
            torch.cuda.FloatTensor(y_train),
            torch.cuda.FloatTensor(c_train_real),
        )
    else:
        train_data = torch.utils.data.TensorDataset(
            torch.cuda.FloatTensor(x_train),
            torch.cuda.FloatTensor(y_train),
        )

    train_dl = torch.utils.data.DataLoader(
        train_data,
        batch_size=experiment_config["batch_size"],
    )

    if load_from_cache and os.path.exists(cem_model_path):
        # Then we simply load the model and proceed
        logging.info("\tFound cached model... loading it")
        cem.load_state_dict(torch.load(cem_model_path))
        cem_time_trained = old_results.get('time_trained')
        cem_epochs_trained = old_results.get('epochs_trained')
    else:

        # Else it is time to train it
        _, cem_time_trained = utils.timeit(
            trainer.fit,
            model=cem,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
        )
        if experiment_config.get("patience", float("inf")) not in [
            None,
            0,
            float("inf"),
        ]:
            cem_epochs_trained = early_stop.stopped_epoch
        else:
            cem_epochs_trained = experiment_config["max_epochs"]
        logging.debug(prefix + "\tCEM training completed")
        logging.debug(prefix + "\tSerializing model")
        torch.save(
            cem.state_dict(),
            cem_model_path,
        )

    end_results['num_params'] = sum([
        p.numel() for p in cem.parameters()
        if p.requires_grad
    ])
    logging.debug(
        prefix +
        f"\tNumber of CEM trainable parameters = {end_results['num_params']}"
    )

    # Log training times and whatnot
    if cem_epochs_trained is not None:
        end_results['epochs_trained'] = cem_epochs_trained
    if cem_time_trained is not None:
        end_results['time_trained'] = cem_time_trained

    # Evaluate our model
    if c_test is not None:
        test_data = torch.utils.data.TensorDataset(
            torch.cuda.FloatTensor(x_test),
            torch.cuda.FloatTensor(y_test),
            torch.cuda.FloatTensor(c_test),
        )
    else:
        test_data = torch.utils.data.TensorDataset(
            torch.cuda.FloatTensor(x_test),
            torch.cuda.FloatTensor(y_test),
        )
    test_dl = torch.utils.data.DataLoader(
        test_data,
        batch_size=experiment_config["batch_size"],
    )

    logging.info(prefix + "\tEvaluating CEM")
    test_output = trainer.predict(cem, test_dl)
    test_concept_probs = np.concatenate(
        list(map(lambda x: x[0], test_output)),
        axis=0,
    )
    test_concept_embeddings = np.concatenate(
        list(map(lambda x: x[1], test_output)),
        axis=0,
    )
    test_output = np.concatenate(
        list(map(lambda x: x[2], test_output)),
        axis=0,
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
                test_concept_probs[:, learnt_concept_idx],
            )
        end_results['avg_concept_auc'] = avg / len(supervised_concept_idxs)
        logging.debug(
            prefix +
            f"\t\tMean Concept AUC is {end_results['avg_concept_auc']*100:.2f}%"
        )


    if (c_train is not None) and (c_test is not None):
        train_concept_probs = trainer.predict(
            cem,
            train_dl,
        )
        train_concept_probs = np.concatenate(
            list(map(lambda x: x[0], train_concept_probs)),
            axis=0,
        )
        logging.debug(prefix + f"\t\tComputing best independent concept aligment...")
        end_results['best_independent_alignment'], end_results['best_ind_alignment_corr'] = \
            metrics.find_best_independent_alignment(
                scores=train_concept_probs,
                c_train=c_train,
            )

        representation_evaluation.evaluate_concept_representations(
            end_results=end_results,
            experiment_config=experiment_config,
            test_concept_scores=test_concept_probs,
            c_test=c_test,
            y_test=y_test,
            old_results=old_results,
            load_from_cache=load_from_cache,
            prefix=prefix,
        )

        if experiment_config.get('perform_interventions', True):
            # Then time to do some interventions!
            logging.debug(prefix + f"\t\tPerforming concept interventions")
            threshs = experiment_config.get(
                'usable_concept_threshold',
                [0.75],
            )
            if not isinstance(threshs, list):
                threshs = [threshs]
            for thresh in threshs:
                selected_concepts = end_results['best_ind_alignment_corr'] >= thresh
                corresponding_real_concepts = np.array(
                    end_results['best_independent_alignment']
                )
                if (
                    (experiment_config.get("n_supervised_concepts", 0) != 0) and
                    (len(experiment_config.get('supervised_concept_idxs', [])) > 0)
                ):
                    # Then compute the mean concept predictive accuracy
                    for learnt_idx, real_idx in enumerate(
                        experiment_config['supervised_concept_idxs']
                    ):
                        selected_concepts[learnt_idx] = True
                        corresponding_real_concepts[learnt_idx] = real_idx

                selected_concepts_idxs = np.array(
                    list(range(experiment_config['n_concepts']))
                )[selected_concepts]
                corresponding_real_concepts = corresponding_real_concepts[selected_concepts]

                end_results[f'interveneable_concepts_{thresh}'] = np.sum(selected_concepts)
                interveneable_concepts = end_results[f'interveneable_concepts_{thresh}']
                logging.debug(
                    prefix + f"\t\t\tNumber of concepts we will intervene on " +
                    f"is {interveneable_concepts}/{experiment_config['n_concepts']}"
                )
                one_hot_labels = tf.keras.utils.to_categorical(y_test)
                for num_intervened_concepts in range(1, interveneable_concepts + 1):
                    def _run():
                        avg = 0.0
                        for i in range(experiment_config.get('intervention_trials', 5)):
                            current_sel = np.random.permutation(
                                list(range(len(selected_concepts_idxs)))
                            )[:num_intervened_concepts]
                            fixed_used_concept_idxs = selected_concepts_idxs[current_sel]
                            real_corr_concept_idx = corresponding_real_concepts[current_sel]
                            intervention_cem = ConceptEmbeddingModel(
                                **cem_params,
                                intervention_idxs=fixed_used_concept_idxs,
                            )
                            intervention_cem.load_state_dict(torch.load(cem_model_path))
                            int_test_output = trainer.predict(intervention_cem, test_dl)
                            int_test_output = np.concatenate(
                                list(map(lambda x: x[2], int_test_output)),
                                axis=0,
                            )
                            avg += sklearn.metrics.accuracy_score(
                                y_test,
                                np.argmax(
                                    scipy.special.softmax(
                                        int_test_output,
                                        axis=-1,
                                    ),
                                    axis=-1
                                ),
                            )
                        return avg / experiment_config.get('intervention_trials', 5)

                    key = f'acc_intervention_{num_intervened_concepts}_thresh_{thresh}'
                    end_results[key] = utils.posible_load(
                        key=key,
                        old_results=old_results,
                        load_from_cache=load_from_cache,
                        run_fn=_run,
                    )
                    logging.debug(
                        prefix +
                        f"\t\t\tIntervention accuracy with {num_intervened_concepts} "
                        f"concepts (thresh = {thresh} with {interveneable_concepts} interveneable concepts): {end_results[key] * 100:.2f}%"
                    )
                    if thresh == threshs[-1]:
                         end_results[f'acc_intervention_{num_intervened_concepts}'] = end_results[key]

    if return_model:
        return end_results, cem
    return end_results
