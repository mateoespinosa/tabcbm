import sklearn
import scipy
import tensorflow as tf
import numpy as np
from collections import defaultdict
import training.train_funcs as train_funcs
import os
import yaml
from pytorch_lightning import seed_everything
import logging
import argparse
import sys
import warnings
import datasets as data
import random
from pathlib import Path

################################################################################
## HELPER FUNCTIONS
################################################################################


def _to_val(x):
    if len(x) >= 2 and (x[0] == "[") and (x[-1] == "]"):
        vals = list(map(lambda x: x.strip(), x[1:-1].split(",")))
        return list(map(_to_val, vals))
    try:
        return int(x)
    except ValueError:
        # Then this is not an int
        pass

    try:
        return float(x)
    except ValueError:
        # Then this is not an float
        pass

    if x.lower() in ["true"]:
        return True
    if x.lower() in ["false"]:
        return False

    return x


def extend_with_global_params(config, global_params):
    for param_path, value in global_params:
        var_names = list(map(lambda x: x.strip(), param_path.split(".")))
        current_obj = config
        for path_entry in var_names[:-1]:
            if path_entry not in config:
                current_obj[path_entry] = {}
            current_obj = current_obj[path_entry]
        current_obj[var_names[-1]] = _to_val(value)

def validate_config(config, required):
    for name in required:
        if name not in config:
            raise ValueError(
                f'Expected required key "{name}" to be given as part of the '
                f'experiment config however we could not find it.'
            )

############################################
## Arg Parser Function
############################################

def build_parser():
    """
    Helper function to build our program's argument parser.

    :returns ArgumentParser: The parser for our program's configuration.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Runs experiments for our tabular architecture.'
        ),
    )
    parser.add_argument(
        '--config',
        '-c',
        default=None,
        help="initial configuration YAML file for our experiment's setup.",
        metavar="file.yaml",
    )
    parser.add_argument(
        '--rerun',
        '-r',
        action='store_true',
        default=False,
        help=(
            "Reruns all experiments even if we find the results cached in the given output "
            "directory."
        ),
    )
    parser.add_argument(
        '--debug',
        '-d',
        action='store_true',
        default=False,
        help=(
            "Debug mode"
        ),
    )
    parser.add_argument(
        '--print_cache_only',
        action='store_true',
        default=False,
        help=(
            "If true then we will simply load and print results from cache without loading a model "
            "recomputing statistics."
        ),
    )
    parser.add_argument(
        '--force_single_process',
        action='store_true',
        default=False,
        help=(
            "If true then all training runs will be done in a single process. This may result in the "
            "GPU becoming overloaded so use with caution."
        ),
    )
    parser.add_argument(
        '--output_dir',
        '-o',
        default=None,
        help=(
            "directory where we will dump our experiment's results. If not "
            "given, then we will use the directory given as the 'results_dir' in "
            "the config file."
        ),
        metavar="path",

    )
    parser.add_argument(
        '--sort_key',
        '-k',
        default="model",
        help=(
            "Field name used to sort the output table for the experiment."
        ),
        metavar="field",

    )
    parser.add_argument(
        '-p',
        '--param',
        action='append',
        nargs=2,
        metavar=('param_name=value'),
        help=(
            'Allows the passing of a config param that will overwrite '
            'anything passed as part of the config file itself.'
        ),
        default=[],
    )
    parser.add_argument(
        '-f',
        '--field_name',
        action='append',
        nargs=2,
        metavar=('field_name_name pretty_name'),
        help=(
            'Include a field as part of the end result table that gets printed out.'
        ),
        default=[],
    )

    return parser

############################################
## Main Function
############################################

def main(
    config_path=None,
    suppress_warnings=True,
    global_params=None,
    output_dir=None,
    load_from_cache=True,
    debug=True,
    result_table_fields=None,
    sort_key="model",
    print_cache_only=False,
    multiprocess_inference=True,
    **kwargs,
):
    
    ############################################################################
    ## Setup
    ############################################################################

    seed_everything(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if suppress_warnings:
        tf.data.experimental.enable_debug_mode()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        warnings.filterwarnings('ignore')
        tf.get_logger().setLevel('FATAL')
        
    

    ############################################################################
    ## Load the config
    ############################################################################

    if config_path is not None:
        if not os.path.exists(config_path):
            raise ValueError(
                f'Given config file "{config_path}" is not a valid YAML file.'
            )
        with open(config_path, 'r') as file:
            experiment_config = yaml.safe_load(file)
    else:
        # else we start with a blank state and all other arguments must be
        # provided through the command line
        experiment_config = {}
        
    # Update it with possible args passed in the command line
    extend_with_global_params(
        experiment_config,
        (global_params or []) + list(kwargs),
    )
    if output_dir is not None:
        if "results_dir" in experiment_config:
            old = experiment_config["results_dir"]
            logging.warning(
                f"Overwritting results directory in given config file from "
                f'"{old}" to "{output_dir}"'
            )
        experiment_config['results_dir'] = output_dir
    
    # Validate result
    validate_config(
        experiment_config,
        required=[
            'dataset',
            'results_dir',
            'runs',
            'trials',
        ],
    )
    logging.info(f"Results will be dumped in {experiment_config['results_dir']}")
    Path(experiment_config['results_dir']).mkdir(parents=True, exist_ok=True)
    # Write down the actual command executed
    with open(os.path.join(experiment_config['results_dir'], "command.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv))
    
    ############################################################################
    ## Load the config
    ############################################################################
    if debug:
        experiment_config['debug_level'] = 'debug'
    debug_level = experiment_config.get('debug_level', 'info').lower().strip()
    if debug_level == "info":
        debug_level = logging.INFO
    elif debug_level == "debug":
        debug_level = logging.DEBUG
    elif debug_level == "fatal":
        debug_level = logging.FATAL
    elif debug_level == "warning":
        debug_level = logging.WARNING
    else:
        used = experiment_config.get('debug_level', 'info')
        raise ValueError(f'Invalid debug level "{used}"')
    print("Using level:", debug_level)
    logger = logging.getLogger()
    logger.setLevel(debug_level)
    logging.basicConfig(
        format='[%(levelname)s] %(message)s'
    )
    
    fh = logging.FileHandler(
        os.path.join(experiment_config['results_dir'], 'output.log')
    )
    fh.setLevel(debug_level)
    logger.addHandler(fh)
    
    ############################################################################
    ## Dataset Generation
    ############################################################################

    ds_name = experiment_config['dataset'].lower().strip()
    if ds_name == "synth_tab_linear":
        data_generator = data.generate_tabular_synth_linear_data
    elif ds_name == "synth_tab_nonlinear":
        data_generator = data.generate_tabular_synth_nonlinear_data
    elif ds_name == "synth_tab_nonlinear_large":
        data_generator = data.generate_tabular_synth_nonlinear_large_data
    elif  ds_name == "synth_sc_data":
        data_generator = data.generate_synth_sc_data
    elif  ds_name == "forest_cover":
        data_generator = data.generate_forest_cover_data
    elif ds_name == "higgs":
        data_generator = data.generate_higgs_data
    elif ds_name == "pbmc":
        data_generator = data.generate_pbmc_data
    else:
        used = experiment_config['dataset']
        raise ValueError(f'Unrecognized dataset name "{used}"')
    
    # Also save a ready-to-use config for recreation purposes
    with open(os.path.join(experiment_config['results_dir'], 'rerun_config.yaml'), "w") as f:
        yaml.dump(experiment_config, f)
    
    ############################################################################
    ## Time to actually run things
    ############################################################################

    table = train_funcs.experiment_loop(
        experiment_config=experiment_config,
        load_from_cache=load_from_cache,
        data_generator=data_generator,
        result_table_fields=result_table_fields,
        sort_key=sort_key,
        print_cache_only=print_cache_only,
        multiprocess_inference=multiprocess_inference,
    )
    return 0

################################################################################
## ENTRY POINT
################################################################################

if __name__ == '__main__':
    # First generate our argument parser
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(main(
        config_path=args.config,
        global_params=args.param,
        load_from_cache=(not args.rerun),
        output_dir=args.output_dir,
        debug=args.debug,
        result_table_fields=args.field_name,
        sort_key=args.sort_key,
        print_cache_only=args.print_cache_only,
        multiprocess_inference=(not args.force_single_process),
    ))
