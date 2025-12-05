# Short-horizon MArket Risk predicting and Trading (SMART)

## Overview

**SMART** is a Python package to implement and evaluate the method described in the [paper](https://arxiv.org/html/2510.22348v1), Causal and Predictive Modeling of Short-Horizon Market Risk and Systematic Alpha Generation Using Hybrid Machine Learning Ensembles, authored by Aryan Ranjan from Oxford University and published on October 25, 2025.


## Installation

- using conda (recommended)

```
git clone git@github.com:HaniceSun/smart.git
cd smart
conda env create -f environment.yml
conda activate smart
```

- using docker

```
git clone git@github.com:HaniceSun/smart.git
cd smart
docker build -t smart:latest .
alias smart="docker run -v $(pwd):/app -it smart:latest smart"
```

- using pip

```
git clone git@github.com:HaniceSun/smart.git
cd smart
pip install .
```

## Quick Start

- smart download

- smart preprocess

- smart train

- smart eval

- smart predict


## Author and License

**Author:** Hance Sun

**Email:** hansun@stanford.edu 

**License:** [MIT License](LICENSE)
