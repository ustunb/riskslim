# risk-slim

RiskSLIM fits simple, customized integer risk scores for binary classification.

[![PyPI](https://img.shields.io/pypi/v/riskslim.svg)](https://pypi.org/project/riskslim/)
[![CI](https://github.com/ustunb/riskslim/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/ustunb/riskslim/actions/workflows/ci.yml?query=branch%3Adev)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)

## What it does

Risk scores let users make quick predictions by adding and subtracting a few small numbers. RiskSLIM searches for sparse integer coefficients while optimizing logistic loss, so the resulting score is easy to inspect and use.

## Install

```bash
pip install riskslim
```

CPLEX Community Edition installs as a required dependency. The Community Edition limits static models to 1,000 variables and 1,000 constraints.

## Quickstart

```python
from pathlib import Path

from riskslim import RiskSLIMClassifier
from riskslim.data import BinaryClassificationDataset, NumericBinarizer

# import data (683 x 9, integer features, outcome in the first column) with 5 CV folds
data_file = Path(__file__).resolve().parents[1] / "data" / "breastcancer_data.csv"
data = BinaryClassificationDataset.read_csv(data_file, n_folds=(5,))

# binarize: each item is a feature at 5 or above (features are 1-10), named "<feature>_geq_5"
data.processor.update({name: NumericBinarizer(thresholds=[5]) for name in data.names.X})
X, y = data.X, data.y
items = list(data.names.X)

# fit a risk score with at most 5 items and coefficients in -5,...,5
clf = RiskSLIMClassifier(
    max_size=5,
    max_coef=5,
    variable_names=items,
    outcome_name=data.names.y,
    cplex_randomseed=0,
    verbose=False,
)
clf.fit(X, y, max_runtime=30.0)

# print the score table
print(clf)

# predict and score
print("train accuracy:", clf.score(X, y))

# cross-validate on the dataset's 5 folds, so the report shows a 5-CV sample next to training
clf.fit_cv(X, y, data=data, max_runtime=30.0)

# save an HTML report of the model
clf.report(model_type="risk_score", data=data).save("riskslim_report.html")
```

The same code is in [`examples/quickstart.py`](examples/quickstart.py), which also fits a checklist on the same items and writes its report.

## Key API

- `RiskSLIMClassifier`: scikit-learn-style estimator with `fit`, `predict`, `score`, and probability methods.
- `RiskSLIMOptimizer`: cutting-plane mixed-integer optimizer for a risk score.
- `CoefficientSet`: integer coefficient bounds and sparsity penalties.
- `riskslim.data.BinaryClassificationDataset`: a binary classification dataset — reads a CSV, binarizes features into items (e.g. `ClumpThickness_geq_5`) and holds the CV folds that `fit_cv` and the report use.
- `RiskSLIMClassifier.report(data=...)`: a `ModelReport` of a fitted model (model, summary table, ROC and calibration plots), with a sample per column (training, 5-CV after `fit_cv`, test); `.save(path)` writes the HTML.

## Paper

Ustun, B. and Rudin, C. “Learning Optimized Risk Scores.” *Journal of Machine Learning Research*, 2019. [Paper](http://jmlr.org/papers/v20/18-615.html) · [BibTeX](https://github.com/ustunb/riskslim/blob/master/docs/references/ustun2019riskslim.bib)

## License and contributing

Released under the [GNU GPL-3.0-or-later](LICENSE). See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.
