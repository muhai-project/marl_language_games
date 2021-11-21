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
        help="optional config file",
        default="cfg/bng.yml",
        type=str,
    )
    args = parser.parse_args()
    return args
