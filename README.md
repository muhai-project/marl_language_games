# Emergent Reinforcement Learning (EMRL)

This repository contains the code for Emergent Reinforcement Learning. It studies emergent communication through the multi-agent reinforcement learning paradigm.

## Installation

Anaconda is probably the fastest way to get up and running with Python environments. However, the full Anaconda toolkit installs 1500 Python packages and takes up 3GB of disk space. A good alternative, therefore, is Miniconda. Miniconda will only install the bare minimum to run Python. It is then up to you to manually install the packages that you _really_ need.

Download Miniconda from [this website](https://docs.conda.io/en/latest/miniconda.html) and follow the installation instructions. If you are prompted to add Miniconda to your `.bash_profile` or `.zshrc`, accept. For the changes to take effect and use `conda` from the command line, you must restart the Terminal. At the time of writing, the Miniconda website suggests to install Miniconda with Python 3.7.

The `environment.yml` file presents in this repository makes it easy to recreate the `emrl` conda environment that was used to develop this project. This file lists all necessary packages and their version numbers. To create a new environment, called `emrl`, from this file, run:

`conda env create -f environment.yml`

If the environment was successfully created, you can activate it by running

`conda activate emrl`

You should see the prompt of your Terminal change from `(base)` to `(emrl)`.
