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

import pandas as pd

from riskslim import RiskSLIMClassifier

# import data (683 x 9, integer features, outcome in the first column)
data_file = Path(__file__).resolve().parents[1] / "data" / "breastcancer_data.csv"
df = pd.read_csv(data_file)
y, X = df.iloc[:, 0], df.iloc[:, 1:]

# fit a risk score with at most 5 variables and coefficients in -5,...,5
clf = RiskSLIMClassifier(
    max_size=5,
    max_coef=5,
    variable_names=list(X.columns),
    outcome_name=df.columns[0],
    cplex_randomseed=0,
    verbose=False,
)
clf.fit(X.values, y.values, max_runtime=30.0)

# print the score table
print(clf)

# predict and score
print("train accuracy:", clf.score(X.values, y.values))

# save an HTML report of the model
clf.reporter.create_report("riskslim_report.html")
```

The same code is in [`examples/quickstart.py`](examples/quickstart.py).

## Key API

- `RiskSLIMClassifier`: scikit-learn-style estimator with `fit`, `predict`, `score`, and probability methods.
- `RiskSLIMOptimizer`: cutting-plane mixed-integer optimizer for a risk score.
- `CoefficientSet`: integer coefficient bounds and sparsity penalties.
- `ClassificationDataset`: binary data wrapper that adds the intercept and maps labels internally.
- `RiskScoreReporter`: score tables and HTML reports for fitted models.

## Paper

Ustun, B. and Rudin, C. “Learning Optimized Risk Scores.” *Journal of Machine Learning Research*, 2019. [Paper](http://jmlr.org/papers/v20/18-615.html) · [BibTeX](https://github.com/ustunb/riskslim/blob/master/docs/references/ustun2019riskslim.bib)

## License and contributing

Released under the [GNU GPL-3.0-or-later](LICENSE). See [CONTRIBUTING.md](CONTRIBUTING.md) for development instructions.
