# AMR Social Chemistry Reasoner

This repo contains code for experiments in the paper "Neuro-Symbolic Commonsense Social Reasoning".

## Setup

This repo manages dependencies using [Poetry](https://python-poetry.org/). After cloning this repo, run `poetry install` to install dependencies.

## Dataset and AMR Parsing

This Repo uses the [Social Chemistry 101](https://maxwellforbes.com/social-chemistry/) dataset. AMR is generated for the rule of thumb and social situation text using the [IBM Transition AMR parser](https://github.com/IBM/transition-amr-parser). IBM's parser isn't published on PyPI, and thus requires manually cloning their repo and requesting access to pretrained models. We include a subset of Social Chemistry 101 data with AMR generated from the IBM AMR Parser in this repo in `data/social_chemistry_101_data_enhanced.json`. If you'd like to regenerate this data yourself, or annotate more Social Chemistry data, there's a helper script under `amr_reasoner/scripts/enhance_social_chemistry_data.py`, (`poetry run python -m amr_reasoner.scripts.enhance_social_chemistry_data`) and a wrapper class for working with IBM's parser under `amr_reasoner.parse.IbmAmrParser`.

## Evaluation

The script `amr_reasoner.scripts.evaluate_social_chemistry_rots` can be used to run evaluation. This script can either be used to perform a grid search over parameter values (by pass the `--grid` option), or can be used to evaluate a specific set of hyperparameters. Both of these are illustrated below:

```bash
# grid search over possible hyperparameter settings
poetry run python -m amr_reasoner.scripts.evaluate_social_chemistry_rots \
    --grid \
    --max-samples 1000

# evaluate a single hyperparameter setting
poetry run python -m amr_reasoner.scripts.evaluate_social_chemistry_rots \
    --min-similarity-threshold 0.9 \
    --max-samples 1000
```

See `amr_reasoner/scripts/evaluate_social_chemistry_rots.py` for a full list of options

## Interactive debugging

To make it easier to interactively explore failed test samples, there's a debugging script which runs evaluation and allows you to interactively explore why the failure occurred, and view intermediate AMR trees, logic, etc for each sample. You can run this with:

```bash
poetry run python -m amr_reasoner.scripts.debug_social_chemistry_evaluation_failures
```

See `amr_reasoner/scripts/debug_social_chemistry_evaluation_failures.py` for a full list of options

## Development and contributing

This repo contains unit tests for core functionality. These can be run with `poetry run pytest`.

In addition, this repo uses [Black](https://black.readthedocs.io/en/stable/) for code formatting, [Flake8](https://flake8.pycqa.org/en/latest/) for linting, and [MyPy](https://mypy.readthedocs.io/en/stable/) for type checking. These can be run as follows:

```bash
poetry run mypy . && poetry run flake8 . && poetry run black .
```
