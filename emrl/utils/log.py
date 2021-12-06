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


def create_logdir(path):
    """Creates a new unique directory for an experiment, returns the path of this newly create folder.

    Args:
        path (str): path where to create the new directory

    Returns:
        str: path of the new directory
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    logdir = os.path.join(path, str(now))
    os.makedirs(logdir, exist_ok=True)
    return logdir


def log_experiment(args, cfg, logdir):
    """Logs the experiment for reproducibility, returns the

    For each experiment a new unique directory is created.
    A copy of the code, config file and the sysout is copied.

    Args:
        args (dict): command-line arguments
        cfg (dict): contains the parameters of the experiment to run
        logdir (str): path of the folder where the experiment is logged
    """
    print(f" === Saving output to: {logdir} === ")

    # copy emrl codebase
    code_dir_name = "emrl"
    code_dir = os.path.join(os.getcwd(), code_dir_name)
    shutil.copytree(
        code_dir,
        os.path.join(logdir, code_dir_name),
        ignore=shutil.ignore_patterns("*.pyc", "tmp*"),
    )

    # copy scripts
    code_dir_name = "scripts"
    code_dir = os.path.join(os.getcwd(), code_dir_name)
    shutil.copytree(
        code_dir,
        os.path.join(logdir, code_dir_name),
        ignore=shutil.ignore_patterns("*.pyc", "tmp*"),
    )

    # copy config file
    cfg_file = args.cfg_file
    shutil.copy(cfg_file, logdir)

    # write terminal output to log
    sys.stdout = Logger(logfile=os.path.join(logdir, "logfile.log"))
    print(" === Using config === ")
    pprint.pprint(vars(args))  # log raw command-line args
    pprint.pprint(cfg)  # log loaded cfg
