# Systematic MARket Trading via machine learning (SMART)

## Overview

*SMART* is a Python package for reproducing and evaluating methods from research papers and online posts, with the recognition that sometimes reported superior results may stem from overfitting or data leakage. Once a method’s performance is validated, the strategy is incorporated into my live trading systems using Backtrader with Alpaca and IBKR brokers.

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
sudo docker build -t smart:latest .
alias smart="sudo docker run -v $(pwd):/app -it smart:latest smart" 
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
