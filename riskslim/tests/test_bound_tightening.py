"""Bound tightening tests."""

import pytest
import numpy as np
from riskslim.bounds import Bounds, chained_updates, \
    chained_updates_for_lp


@pytest.mark.parametrize('update', [chained_updates, chained_updates_for_lp])
@pytest.mark.parametrize('loss', [(10, 1e-6), (1e-6, 10)])
@pytest.mark.parametrize('new_objvals', [(10, 1e-6), (1e-6, 10)])
@pytest.mark.parametrize('C_0_nnz', [.01, 10])
def test_chained_updates(update, loss, new_objvals, C_0_nnz):

    bounds = Bounds(
        objval_min=1,
        objval_max=10,
        loss_min=loss[0],
        loss_max=loss[1],
        min_size=1,
        max_size=10
    )

    new_objval_at_feasible, new_objval_at_relaxation = new_objvals

    new_bounds = update(
        bounds, np.tile(C_0_nnz, 100), new_objval_at_feasible, new_objval_at_relaxation
    )

    assert new_bounds.objval_min >= bounds.objval_min
    assert new_bounds.objval_max <= bounds.objval_max

    assert new_bounds.loss_min >= bounds.loss_min
    assert new_bounds.loss_max <= bounds.loss_max

    assert new_bounds.min_size >= bounds.min_size
    assert new_bounds.max_size <= bounds.max_size
