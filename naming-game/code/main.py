import argparse

from plot import average_sliding_window
from experiment import Experiment
from log import Logger
from utils import cfg_from_file, log_experiment

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='cfg/default.yml', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg = cfg_from_file(args.cfg_file)
    logdir = log_experiment(args, cfg)
    experiment = Experiment(cfg)
    experiment.run_experiment()
