import time
import tensorflow as tf
import numpy as np
import os
import random
import warnings
import logging
import nvidia_smi
import pytorch_lightning

############################################
## Utils
############################################

def print_gpu_usage():
    nvidia_smi.nvmlInit()
    for i in range(nvidia_smi.nvmlDeviceGetCount()):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        logging.debug(
            f"\tGPU device {i}: {nvidia_smi.nvmlDeviceGetName(handle)}, "
            f"Memory : ({100*info.free/info.total:.2f}% free): "
            f"{info.total}(total), {info.free} (free), {info.used} (used)"
        )
    nvidia_smi.nvmlShutdown()
    
def posible_load(
    key,
    run_fn,
    old_results,
    load_from_cache=True,
):
    keys = key
    if not isinstance(keys, (list, tuple)):
        keys = [key]
        
    if old_results and load_from_cache:
        result = []
        for k in keys:
            if k in old_results:
                result.append(old_results[k])
            else:
                break
        if len(result) == len(keys):
            return result[0] if len(keys) == 1 else tuple(result)
    return run_fn()

def timeit(f, *args, **kwargs):
    start = time.time()
    result = f(*args, **kwargs)
    end = time.time()
    return result, (end - start)

def restart_seeds(trial=0):
    pytorch_lightning.utilities.seed.seed_everything(42 + trial)
    os.environ['PYTHONHASHSEED'] = str(42 + trial)
    tf.random.set_seed(42 + trial)
    np.random.seed(42 + trial)
    random.seed(42 + trial)
    # And also let's reduce the noise from warnings
    warnings.filterwarnings('ignore')
    
    # Reset the logging in case we are using a subprocess
    print("\tSetting log level to:", os.environ.get('LOGLEVEL', 'WARNING').upper())
    logging.getLogger().setLevel(
        os.environ.get('LOGLEVEL', 'WARNING').upper()
    )
