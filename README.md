# Multi-agent Reinforcement Learning (MARL) Language Games

This repository accompanies the paper [Language games meet multi-agent reinforcement learning: A case study for the naming game](https://academic.oup.com/jole/article/7/2/213/7128304#410601591). It studies emergent communication through the multi-agent reinforcement learning framework and the language games paradigm.

## Project Organization

    ├── cfg                <- Configurations files for experiments
    │
    ├── data               <- The original, immutable logged experiments.
    │
    ├── marl_language_games <- Source code for use in this project.
    │
    ├── scripts            <- Scripts of the marl_language_games package
    │
    ├── tests              <- Unit tests for marl_language_games package
    │
    ├── Makefile           <- Makefile with commands to create conda env
    │
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── environment.yml    <- The project's package dependency list for reproducing the environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so marl_language_games can be imported
    │
    └── setup.cfg          <- pytest, flake8, black and isort settings

## Installation

Anaconda is probably the fastest way to get up and running with Python environments. However, the full Anaconda toolkit installs 1500 Python packages and takes up 3GB of disk space. A good alternative, therefore, is Miniconda. Miniconda will only install the bare minimum to run Python. It is then up to you to manually install the packages that you _really_ need.

Download Miniconda from [this website](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions. If you are prompted to add Miniconda to your `.bash_profile` or `.zshrc`, accept. For the changes to take effect and use `conda` from the command line, you must restart the Terminal. At the time of writing, the Miniconda website suggests to install Miniconda with Python 3.7.

The `environment.yml` file presents in this repository makes it easy to recreate the `marl_language_games` conda environment that was used to develop this project. This file lists all necessary packages and their version numbers. You can use the Makefile to setup this environment.

1. To recreate the `marl_language_games` conda environment that was used to develop this project, run:
   - `make install_conda_env`
2. If the environment was successfully created, you can activate it by running
   - `conda activate marl_language_games`
   - You should see the prompt of your Terminal change from `(base)` to `(marl_language_games)`.
3. Then to install the `marl_language_games` package into the newly create `(marl_language_games)` environment, run:
   - `make install_package`
   - You should see the `marl_language_games` package when listing all dependencies of the environment through `conda list`.

## Running an experiment

Make sure that the `marl_language_games` environment has been setup and activated.

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
- `--debug`
  - [optional] [flag] [default: `false`]
  - specify whether to log DEBUG-level messages
  - messages are logged to a logfile in the unique directory under `--log_path`
- `--print-every`
  - [optional] [int] [default `1000`]
  - requires `--debug` flag to be set
  - logs every x-th communicative interaction (and prints to stdout)

For example, the following command runs the basic naming game experiment with the parameters specified in the configuration file found at `cfg/config.yml`.

```
python scripts/run_experiment.py --cfg cfg/config.yml --debug --print_every 5000
```

## Generate plots

Once the experiments have completed, plots can be generated for the logged experiments. The Babel library contains an extensive and powerful plot engine. The engine requires the data of the experiments to be in a particular format, therefore in `utils/convert_data.py` the logger formats the logged data into the format expected by `Babel`. The script to produce the figures of the paper can be found under [`/experiments/emergent-rl/`](https://gitlab.ai.vub.ac.be/ehai/ehai-babel/-/blob/master/experiments/emergent-rl/) in `run.lisp`.

Alternatively, we also provide a way to generate the plots directly without Babel. This feature is available by calling the `plot-monitors` in `utils/plot.py` with the list of monitors.

## Unit tests

This repository provides unit tests (with pytest) for the `marl_language_games` package in the `tests/` folder. The conda environment associated with the `environment.yml` installs `pytest`. The tests can be run with `pytest` in the command-line.

## Citation

```
@article{vaneecke2022language,
    title = {Language games meet multi-agent reinforcement learning: A case study for the naming game},
    author = {{Van Eecke}, Paul and Beuls, Katrien and {Botoko Ekila}, J\'{e}r\^{o}me and R{\u{a}}dulescu, Roxana},
    year = {2022},
    journal = {Journal of Language Evolution},
    volume = {7},
    number = {2},
    pages = {213--223},
    doi = {10.1093/jole/lzad001},
}
```

## Acknowledgements

The research reported on in this paper was financed by the Research Foundation Flanders (FWO - Vlaanderen) through postdoctoral grants awarded to Paul Van Eecke (75929) and Roxana Rădulescu (1286223N), and by the European Union’s Horizon 2020 research and innovation programme under grant agreement no. 951846 (MUHAI - https://www.muhai.org).
