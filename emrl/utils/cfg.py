import argparse

import yaml
from easydict import EasyDict as edict


def cfg_from_file(filename):
    """Loads a yaml config file."""
    with open(filename, "r") as f:
        yaml_cfg = edict(yaml.safe_load(f))
    return yaml_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="config file of the experiment",
        required=True,
        type=str,
        nargs="*",
    )
    parser.add_argument(
        "--log_path",
        dest="log_path",
        help="path to the directory where to log the experiment",
        type=str,
        default="data/log",
    )
    parser.add_argument(
        "--debug", dest="debug", help="activates debug logging", action="store_true"
    )
    parser.add_argument(
        "--print_every",
        dest="print_every",
        help="print every x iterations an example interaction",
        type=int,
        default=1000,
    )
    args = parser.parse_args()
    return args
