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
"""RiskSLIM quickstart — deterministic, no network, CE-sized."""

import numpy as np
from sklearn.datasets import load_breast_cancer  # sklearn is already a runtime dependency

from riskslim import RiskSLIMClassifier


def main():
    X, y = load_breast_cancer(return_X_y=True)  # 569 x 30, bundled with sklearn
    X = (X > np.median(X, axis=0)).astype(int)  # binarize at medians -> 30 binary features
    names = [f"{n} > median" for n in load_breast_cancer().feature_names]
    clf = RiskSLIMClassifier(
        max_coef=5,
        max_size=5,
        variable_names=names,
        outcome_name="malignant",
        verbose=False,
        cplex_randomseed=0,
        initialization_flag=True,
        init_max_runtime=5.0,
        init_use_rounding=False,
        init_use_sequential_rounding=False,
        init_polishing_after=False,
    )
    clf.fit(X, y, max_runtime=30.0)  # y in {0,1} must work (defect m)
    print(clf)  # score table
    print("train accuracy:", clf.score(X, y))
    clf.reporter.create_report("riskslim_report.html")


if __name__ == "__main__":
    main()
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
