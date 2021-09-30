from collections import defaultdict
import numpy as np
import yaml
import datetime
import dateutil
import dateutil.tz
import os
import shutil
import yaml
import pprint
from easydict import EasyDict as edict

from log import Logger
import sys

### Language helper functions
#################################

ids = defaultdict(int)
def make_id(name):
    """
    Creates a unique id for an object resembling a lisp symbol.

    Each input gets its own counter that is incremented each time the function is called.
    To use:
    >>> make_id("AG")
    #'AG-0
    >>> make_id("AG")
    #'AG-1
    >>> make_id("OBJ")
    #'OBJ-0
    >>> make_id("AG")
    #'AG-2
    >>> make_id("OBJ")
    #'OBJ-1
    
    Args:
        name: A string for which a so-called symbolic id is created.

    Returns:
        A string that is unique each time the function is called.
    """
    global ids
    val = f"#'{name}-{ids[name]}"
    ids[name] += 1
    return val

def invent(syllables=3):
    """
    Invents a word with a number of syllables through random sampling of letters.
    
    Each syllable has exactly two letters: a consonant and a vowel (in that order).
    The two letters are composed by randomly sampling from the set of possible choices.

    Args:
        syllables: An integer representing the amount of syllables in the new word.

    Returns:
        A string that is randomly generated.
    """
    vowels = ["a", "e", "i", "o", "u"]
    consonants = ["b", "c", "d", "f", "g", "h", "j", 
                  "k", "l","m", "n", "p", "q", "r", 
                  "s", "t", "v", "w", "x", "y", "z"]
    
    # produce word containing a sequence of syllables
    new_word = ""
    for i in range(syllables):
        new_word += np.random.choice(consonants) + np.random.choice(vowels)
    return new_word

### Loading
#################


def cfg_from_file(filename):
    """Loads a yaml config file."""
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.safe_load(f))
    return yaml_cfg

### Logging
#################

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
    logdir = f"data/{now}"
    os.makedirs(logdir, exist_ok=True)
    print(f" === Saving output to: {logdir} === ")

    # copy codebase
    code_dir = os.path.join(os.getcwd(), "code")
    os.makedirs(os.path.join(logdir, "code"), exist_ok=True)
    for filename in os.listdir(code_dir):
        if filename.endswith(".py"):
            shutil.copy(code_dir + "/" + filename, os.path.join(logdir, "code"))

    # copy config file
    shutil.copy(cfg_file, logdir)

    # write terminal output to log
    sys.stdout = Logger(logfile=os.path.join(logdir, "logfile.log"))
    print(' === Using config === ')
    pprint.pprint(vars(args)) # log raw command-line args
    pprint.pprint(cfg) # log loaded cfg

    return logdir
