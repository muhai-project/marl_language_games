# Emergent Reinforcement Learning (EMRL)

This repository contains the code for Emergent Reinforcement Learning. It studies emergent communication through the multi-agent reinforcement learning paradigm.

## Project Organization

    ├── cfg                <- Configurations files for experiments
    │
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- Place logged experiments .
    │   └── log            <- The original, immutable logged experiments.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org
    │
    ├── emrl               <- Source code for use in this project.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── scripts            <- Scripts of the emrl package
    │
    ├── tests              <- Unit tests for emrl package
    │
    ├── Makefile           <- Makefile with commands to create conda env
    │
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── environment.yml    <- The project's package dependency list for reproducing the environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so emrl can be imported
    │
    └── setup.cfg          <- pytest, flake8, black and isort settings

## Installation

Anaconda is probably the fastest way to get up and running with Python environments. However, the full Anaconda toolkit installs 1500 Python packages and takes up 3GB of disk space. A good alternative, therefore, is Miniconda. Miniconda will only install the bare minimum to run Python. It is then up to you to manually install the packages that you _really_ need.

Download Miniconda from [this website](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions. If you are prompted to add Miniconda to your `.bash_profile` or `.zshrc`, accept. For the changes to take effect and use `conda` from the command line, you must restart the Terminal. At the time of writing, the Miniconda website suggests to install Miniconda with Python 3.7.

The `environment.yml` file presents in this repository makes it easy to recreate the `emrl` conda environment that was used to develop this project. This file lists all necessary packages and their version numbers. You can use the Makefile to setup this environment.

1. To recreate the `emrl` conda environment that was used to develop this project, run:
   - `make install_env1`
2. If the environment was successfully created, you can activate it by running
   - `conda activate emrl`
   - You should see the prompt of your Terminal change from `(base)` to `(emrl)`.
3. Then to install the `emrl` package into the newly create `(emrl)` environment, run:
   - `make install_env2`
   - You should see the `emrl` package when listing all dependencies of the environment through `conda list`.

## Running an experiment

Make sure that the `emrl` environment has been setup and activated.

Two scripts are available in the directory `scripts` at the moment:

```
run_experiment.py # running a full experiment with multiple trials
run_competition.py # running an experiment solely for the purpose of creating competition graphs
```

Both scripts allow the following command-line args:

- `--cfg`
  - [required] [str]
  - specifies a path to a yml config file
  - examples of configs can be found in the `cfg/` directory
- `--log_path`
  - [optional] [str] [default: `'/data/log'`]
  - specify the path where to store the experiment
  - each run is stored in a unique directory in `--log_path`
- `--debug`
  - [optional] [flag] [default: `false`]
  - specify whether to log DEBUG-level messages
  - messages are logged to a logfile in the unique directory under `--log_path`
- `--print-every`
  - [optional] [int] [default `1000`]
  - requires `--debug` flag to be set
  - logs every x-th communicative interaction (and prints to stdout)

For example, the following command runs the basic naming game experiment with the parameters specified in the configuration file found at `cfg/bng.yml`.

```
python scripts/run_experiment.py --cfg cfg/bng.yml
```

## How to generate plots

Once the experiments have completed, plots can be generated for the logged experiments. The Babel library contains an extensive and powerful plot engine. The engine requires the data of the experiments to be in a particular format, therefore in `utils/convert_data.py` the logger formats the logged data into the format expected by `Babel`. The script to produce the figures of the paper can be found under [`/experiments/emergent-rl/`](https://gitlab.ai.vub.ac.be/ehai/ehai-babel/-/blob/master/experiments/emergent-rl/) in `run.lisp`.
