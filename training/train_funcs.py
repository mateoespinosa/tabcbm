import tensorflow as tf
import numpy as np
import os
import joblib
import torch
import copy
import multiprocessing
import gc
import logging
import itertools

from collections import defaultdict
from pathlib import Path
from prettytable import PrettyTable

from training.train_ccd import train_ccd
from training.train_gbm import train_xgboost, train_lightgbm
from training.train_mlp import train_mlp
from training.train_senn import train_senn
from training.train_tabcbm import train_tabcbm
from training.train_cbm import train_cbm
from training.train_tabnet import train_tabnet
from training.train_pca import train_pca
from training.train_cem import train_cem
from training.train_tabtransformer import train_tabtransformer
import training.utils as utils

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

def _generate_hyperatemer_configs(config):
    if "grid_variables" not in config:
        # Then nothing to see here so we will return
        # a singleton set with this config in it
        return [config]
    # Else time to do some hyperparameter search in here!
    vars = config["grid_variables"]
    options = []
    for var in vars:
        if var not in config:
            raise ValueError(
                f'All variable names in "grid_variables" must be exhisting '
                f'fields in the config. However, we could not find any field with '
                f'name "{var}".'
            )
        if not isinstance(config[var], list):
            raise ValueError(
                f'If we are doing a hyperparamter search over variable '
                f'"{var}", we expect it to be a list of values. Instead '
                f'we got {config[var]}.'
            )
        options.append(config[var])
    mode = config.get('grid_search_mode', "exhaustive").lower().strip()
    if mode in ["grid", "exhaustive"]:
        iterator = itertools.product(*options)
    elif mode in ["paired"]:
        iterator = zip(*options)
    else:
        raise ValueError(
            f'The only supported values for grid_search_mode '
            f'are "paired" and "exhaustive". We got {mode} '
            f'instead.'
        )
    result = []
    for specific_vals in iterator:
        current = copy.deepcopy(config)
        for var_name, new_val in zip(vars, specific_vals):
            current[var_name] = new_val
        result.append(current)
    return result
                

############################################
## Main Experiment Loop
############################################

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
    cat_features_fn=None,
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
        "Corr-Aligned Concept Mask AUC",
        "Corr-Aligned Concept AUC",
        "Best Mean Concept Mask AUC",
        "Best Mean Concept AUC",
        "Feature Importance Diff",
        "Feature Selection AUC",
    ]
    result_table_fields_keys = [
        "acc",
        "cas",
        "corr_aligned_mean_mask_auc",
        "corr_aligned_concept_auc",
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

            
        logging.debug(f"\tx_train shape is {x_train.shape} and type is {x_train.dtype}")
        logging.debug(f"\tx_test shape is {x_test.shape} and type is {x_test.dtype}")
        logging.debug(f"\ty_train shape is {y_train.shape} and type is {y_train.dtype}")
        logging.debug(f"\ty_test shape is {y_test.shape} and type is {y_test.dtype}")
        if c_train is not None:
            logging.debug(f"\tc_train shape is {c_train.shape} and type is {c_train.dtype}")
        if c_test is not None:
            logging.debug(f"\tc_test shape is {c_test.shape} and type is {c_test.dtype}")
        logging.info(
            f"\tTrain class distribution: {np.mean(tf.keras.utils.to_categorical(y_train), axis=0)}"
        )
        logging.info(
            f"\tTest class distribution: {np.mean(tf.keras.utils.to_categorical(y_test), axis=0)}"
        )
        if c_train is not None:
            logging.info(
                f"\tTrain concept distribution: {np.mean(c_train, axis=0)}"
            )
        if c_test is not None:
            logging.info(
                f"\tTest concept distribution: {np.mean(c_test, axis=0)}"
            )
        
        # See if there are any dimensions that we know are
        # categorical
        if cat_features_fn is not None:
            cat_feat_inds, cat_dims = cat_features_fn(
                **experiment_config.get('data_hyperparams', {})
            )
            logging.debug(f"\tcat_dims: {cat_feat_inds}")
        else:
            cat_feat_inds, cat_dims = None, None
        for current_config in experiment_config['runs']:
            if restart_gpu_on_run_trial:
                device = cuda.get_current_device()
                device.reset()
            # Construct the config for this particular trial
            trial_config = copy.deepcopy(experiment_config.get('shared_params', {}))
            trial_config['input_shape'] = trial_config.get(
                'input_shape',
                x_train.shape[1:],
            )
            logging.debug(f"\tInput shape is: {trial_config['input_shape']}")
            if c_train is not None:
                trial_config['n_ground_truth_concepts'] = trial_config.get(
                    'n_ground_truth_concepts',
                    c_train.shape[-1],
                )
                logging.debug(
                    f"\tNumber of ground truth concepts "
                    f"is: {trial_config['n_ground_truth_concepts']}"
                )
            trial_config['num_outputs'] = trial_config.get(
                'num_outputs',
                len(set(y_train)) if len(set(y_train)) > 2 else 1,
            )
            logging.debug(f"\tNumber of outputs is: {trial_config['num_outputs']}")
            
                                              
            trial_config.update(current_config)
            trial_config.update(extra_hypers)
            # Now time to iterate over all hyperparameters that were given as part
            for run_config in _generate_hyperatemer_configs(trial_config):
                _evaluate_expressions(run_config)
                # Find the model which we will be using
                arch = run_config['model']
                arch_name = arch.lower().strip()
                cast_fn = lambda x: x
                if arch_name == 'tabcbm':
                    train_fn = train_tabcbm
                    cast_fn = lambda x: x.astype(np.float32)
                    extra_kwargs = dict(
                        cov_mat=cov_mat,
                        ground_truth_concept_masks=ground_truth_concept_masks,
                        cat_feat_inds=cat_feat_inds,
                        cat_dims=cat_dims,
                    )
                elif arch_name == 'cbm':
                    train_fn = train_cbm
                    cast_fn = lambda x: x.astype(np.float32)
                    extra_kwargs = {}
                elif arch_name == 'cem':
                    train_fn = train_cem
                    cast_fn = lambda x: x.astype(np.float32)
                    extra_kwargs = {}
                elif arch_name == "ccd":
                    train_fn = train_ccd
                    cast_fn = lambda x: x.astype(np.float32)
                    extra_kwargs = dict(
                        cat_feat_inds=cat_feat_inds,
                        cat_dims=cat_dims,
                    )
                elif arch_name == "xgboost":
                    train_fn = train_xgboost
                    extra_kwargs = dict(
                        ground_truth_concept_masks=ground_truth_concept_masks,
                        cat_feat_inds=cat_feat_inds,
                        cat_dims=cat_dims,
                    )
                elif arch_name == "lightgbm":
                    train_fn = train_lightgbm
                    extra_kwargs = dict(
                        ground_truth_concept_masks=ground_truth_concept_masks,
                        cat_feat_inds=cat_feat_inds,
                        cat_dims=cat_dims,
                    )
                elif arch_name == "pca":
                    train_fn = train_pca
                    extra_kwargs = dict()
                elif arch_name == "tabnet":
                    train_fn = train_tabnet
                    cast_fn = lambda x: x.astype(np.float32)
                    extra_kwargs = dict(
                        ground_truth_concept_masks=ground_truth_concept_masks,
#                         cat_feat_inds=cat_feat_inds,
#                         cat_dims=cat_dims,
                    )
                elif arch_name == "tabtransformer":
                    train_fn = train_tabtransformer
                    cast_fn = lambda x: x.astype(np.float32)
                    extra_kwargs = dict(
                        ground_truth_concept_masks=ground_truth_concept_masks,
                        cat_feat_inds=cat_feat_inds,
                        cat_dims=cat_dims,
                    )
                elif arch_name == "mlp":
                    train_fn = train_mlp
                    cast_fn = lambda x: x.astype(np.float32)
                    extra_kwargs = dict(
                        cat_feat_inds=cat_feat_inds,
                        cat_dims=cat_dims,
                    )
                elif arch_name == "senn":
                    train_fn = train_senn
                    cast_fn = lambda x: x.astype(np.float32)
                    extra_kwargs = dict(
#                         cat_feat_inds=cat_feat_inds,
#                         cat_dims=cat_dims,
                    )
                else:
                    raise ValueError(f'Unsupported model architecture "{arch}"')
                if x_train is not None:
                    x_train = cast_fn(x_train)
                if x_test is not None:
                    x_test = cast_fn(x_test)

                # Set up[ a local directory for this model to use for its results
                run_config["results_dir"] = os.path.join(base_results_dir, arch)
                initialize_result_directory(run_config["results_dir"])

                
                # Now time to actually train things and see what comes out
                # of this
                extra_name = run_config.get('extra_name', "").format(**run_config)
                if extra_name:
                    extra_name = "_" + extra_name
                logging.info(
                    f"\tRunning Trial {trial + 1}/{experiment_config['trials']} "
                    f"for {arch}{extra_name}:"
                )
                utils.print_gpu_usage()
                
                # Serialize the configuration we will be using for these experiments
                joblib.dump(
                    run_config,
                    os.path.join(run_config['results_dir'], f"config{extra_name + f'_trial_{trial}'}.joblib"),
                )

                local_results_path = os.path.join(
                    run_config["results_dir"],
                    f"results{extra_name + f'_trial_{trial}'}.joblib",
                )
                old_results = None
                if os.path.exists(local_results_path):
                    old_results = joblib.load(local_results_path)

                if (run_config.get('n_supervised_concepts', 0) > 0) and (c_train is not None):
                    logging.debug(
                        f"We will provide supervision "
                        f"for {run_config.get('n_supervised_concepts')} concepts"
                    )
                    if (load_from_cache and (not run_config.get('force_rerun', False))) and (
                        'supervised_concept_idxs' in (old_results or {})
                    ):
                        supervised_concept_idxs = old_results['supervised_concept_idxs']
                    else:
                        np.random.seed(trial + run_config.get('seed', 0))
                        concept_idxs = np.random.permutation(
                            list(range(run_config['n_ground_truth_concepts']))
                        )
                        concept_idxs = sorted(concept_idxs[:run_config['n_supervised_concepts']])
                    run_config['supervised_concept_idxs'] = concept_idxs
                    logging.debug(f"\tSupervising on concepts {concept_idxs}")

                if print_cache_only and (load_from_cache and (not run_config.get('force_rerun', False))) and (
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
                        load_from_cache=load_from_cache and (not run_config.get('force_rerun', False)),
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
                            load_from_cache=load_from_cache and (not run_config.get('force_rerun', False)),
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
                serialized_trial_results = {}
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

