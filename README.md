# Systematic MARket Trading via machine learning (SMART)

## Overview

**SMART** is a Python package for reproducing and evaluating methods shared from papers and posts, with the recognition that sometimes reported superior results may stem from overfitting or data leakage. Once a method’s performance is validated, the strategy is incorporated into my live trading systems using Backtrader with Alpaca and IBKR brokers.

## Installation

- using conda (recommended)

```
git clone git@github.com:HaniceSun/smart.git
cd smart
conda env create -f environment.yml
conda activate smart
```

- using pip

```
git clone git@github.com:HaniceSun/smart.git
cd smart
pip install .
```

- using docker

```
git clone git@github.com:HaniceSun/smart.git
cd smart
sudo docker build -t smart:latest .
alias smart="sudo docker run -v $(pwd):/app -it smart:latest smart" 
```

## Quick Start

- smart download

- smart preprocess

- smart train

- smart eval

- smart predict

## Growing list of implemented studies

The StudyID shown in brackets is needed for command-line usage (e.g. smart download --study ranjan2025)

- [Causal and Predictive Modeling of Short-Horizon Market Risk and Systematic Alpha Generation Using Hybrid Machine Learning Ensembles](https://arxiv.org/abs/2510.22348), Aryan Ranjan, 2025 (ranjan2025)

- [Statistical Arbitrage: Can we get over 20% of annual returns uncorrelated with market risk?](https://www.quantitativo.com/p/statistical-arbitrage?r=2cqd5p), Quantitativo, 2024 (quantitativo2024)


## Author and License

**Author:** Hance Sun

**Email:** hansun@stanford.edu 

**License:** [MIT License](LICENSE)
