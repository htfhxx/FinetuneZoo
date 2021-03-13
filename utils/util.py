import os
import time
import numpy as np
import random
import torch


def logging_file(file):
    def write_log(s):
        print(s, flush=True)
        with open(file, 'a') as f:
            f.write(s+'\n')

    return write_log

def set_up_logging(config):
    if not os.path.exists(config['log_path']):
        os.mkdir(config['log_path'])
    tmp_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()).replace(':','_')
    if config['save_dir_name']=='balalala':
        log_path = config["log_path"] + tmp_time + '/'
        checkpoint_dir = config["checkpoint_dir"] + tmp_time + '/'
    else:
        log_path = config["log_path"] + config['save_dir_name'] + '/'
        checkpoint_dir = config["checkpoint_dir"] + config['save_dir_name'] + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    logging = logging_file(log_path + 'log.txt')  # 往这个文件里写记录
    logging("__________logging the args_________")
    for k, v in config.items():
        logging("%s:\t%s" % (str(k), str(v)))
    logging("\n")
    return logging,log_path,checkpoint_dir

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf``
    (if installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)