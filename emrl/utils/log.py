import datetime
import logging
import os
import shutil
import sys
from pprint import pformat

import dateutil
import dateutil.tz


class Logger(object):
    def __init__(self, logfile, mode):
        logging.basicConfig(
            filename=logfile,
            filemode="a",
            format="%(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
            level=mode,
        )

        self.root = logging.getLogger()
        self.root.setLevel(mode)

        self.handler = logging.StreamHandler(sys.stdout)
        self.handler.setLevel(mode)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        self.handler.setFormatter(formatter)
        self.root.addHandler(self.handler)

    def close(self):
        self.handler.flush = sys.stdout.flush
        self.root.removeHandler(self.handler)
        self.root.handlers.clear()


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


def log_experiment(args, cfg_file, cfg, logdir):
    """Logs the experiment for reproducibility, returns the

    For each experiment a new unique directory is created.
    A copy of the code, config file and the sysout is copied.

    Args:
        args (dict): command-line arguments
        cfg (dict): contains the parameters of the experiment to run
        logdir (str): path of the folder where the experiment is logged

    Returns:
        Logger: handles logging administration
    """
    # set up logger
    logger = Logger(
        logfile=os.path.join(logdir, "logfile.log"),
        mode=(logging.DEBUG if args.debug else logging.INFO),
    )
    logging.info(f" === Saving output to: {logdir} === ")
    logging.info(" === Using config === ")
    logging.info(pformat(vars(args)))  # log raw command-line args
    logging.info(f" this experiment uses cfg file: {cfg_file}")
    logging.info(pformat(cfg))  # log loaded cfg

    # copy emrl codebase
    code_dir_name = "emrl"
    code_dir = os.path.join(os.getcwd(), code_dir_name)
    shutil.copytree(
        code_dir,
        os.path.join(logdir, code_dir_name),
        ignore=shutil.ignore_patterns("*.pyc", "tmp*"),
    )
    shutil.make_archive(
        os.path.join(logdir, "emrl"), "zip", os.path.join(logdir, code_dir_name)
    )
    shutil.rmtree(os.path.join(logdir, code_dir_name))

    # copy scripts
    code_dir_name = "scripts"
    code_dir = os.path.join(os.getcwd(), code_dir_name)
    shutil.copytree(
        code_dir,
        os.path.join(logdir, code_dir_name),
        ignore=shutil.ignore_patterns("*.pyc", "tmp*"),
    )
    shutil.make_archive(
        os.path.join(logdir, "scripts"), "zip", os.path.join(logdir, code_dir_name)
    )
    shutil.rmtree(os.path.join(logdir, code_dir_name))

    # copy config file
    shutil.copy(cfg_file, logdir)

    return logger
