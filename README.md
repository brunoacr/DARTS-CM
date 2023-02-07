# DARTS-CM

This code accompanies the paper "Towards Explaining Actions of Agents using Neural Architecture Search", where this implementation of [Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) is used to learn [Concept Mapping](https://ojs.aaai.org/index.php/AAAI/article/view/16626) architectures.


## Requirements
`tensorflow-gpu=2.5
dill
pandas
python-graphviz
pydot
pillow
huggingface_hub`


If you're using Anaconda, you can download the "dartscm-env.yml" file and use `conda env create -f dartscm-env.yml` to install all packages into a new environment.

## Usage

To reproduce the paper's experiments, run [run_tests.py](run_tests.py).
To run a custom search, run [train_search.py](train_search.py)
