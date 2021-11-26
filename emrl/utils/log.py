import datetime
import os
import pprint
import shutil
import sys

import dateutil
import dateutil.tz


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        sys.stdout.flush()
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def log_experiment(args, cfg):
    """
    Logs the experiment for reproducibility.

    For each experiment a new unique directory is created.
    A copy of the code, config file and the sysout is copied.

    Args:
        cfg_file: A string denoting the path to the config file.
    """
    # create unique directory
    cfg_file = args.cfg_file
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    logdir = f"data/log/{now}"
    os.makedirs(logdir, exist_ok=True)
    print(f" === Saving output to: {logdir} === ")

    # copy codebase
    code_dir_name = "emrl"
    code_dir = os.path.join(os.getcwd(), code_dir_name)
    shutil.copytree(
        code_dir,
        os.path.join(logdir, code_dir_name),
        ignore=shutil.ignore_patterns("*.pyc", "tmp*"),
    )

    # copy config file
    shutil.copy(cfg_file, logdir)

    # write terminal output to log
    sys.stdout = Logger(logfile=os.path.join(logdir, "logfile.log"))
    print(" === Using config === ")
    pprint.pprint(vars(args))  # log raw command-line args
    pprint.pprint(cfg)  # log loaded cfg

    return logdir
