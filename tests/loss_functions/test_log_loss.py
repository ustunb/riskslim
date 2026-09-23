"""Tests for log loss functions."""

import pytest
import numpy as np

from riskslim.loss_functions import log_loss
@pytest.mark.parametrize('lf', [log_loss])
def test_log_loss_value(lf):
    """Test log loss accuracy for small weights."""
    for d in range(1, 101):

        Z = np.ones((d, d), dtype=np.float64, order='F')
        r = np.random.uniform(-1, 1)
        weights = np.ones(d, order='F') + r

        np.testing.assert_approx_equal(
            lf.log_loss_value(Z, weights), np.log1p(np.exp(-(1+r) * d))
        )


@pytest.mark.parametrize('lf', [log_loss])
def test_log_loss_value_and_slope(generated_normal_data, lf):
    """Test accuracy of the direction of log loss slope."""
    # Get simulated data from fixture
    Z = generated_normal_data['Z'][0]
    weights = generated_normal_data['weights']

    # Loss
    loss, slope = lf.log_loss_value_and_slope(Z, weights)

    # Step
    step_size = .1
    loss_step = log_loss.log_loss_value(Z, weights-step_size*slope)

    assert loss_step < loss


@pytest.mark.parametrize('lf', [log_loss])
def test_log_loss_value_from_scores(generated_normal_data, lf):
    """Test log loss from scores."""
    # Get simulated data from fixture
    Z = generated_normal_data['Z'][0]
    weights = generated_normal_data['weights']

    # Ensure implementation is the same between functions
    assert lf.log_loss_value_from_scores(Z.dot(weights)) == lf.log_loss_value(Z, weights)


def test_log_probs(generated_normal_data):
    """Test log probabiltiies."""
    # Get simulated data from fixture
    Z = generated_normal_data['Z'][0]
    X = generated_normal_data['X'][0]
    weights = generated_normal_data['weights_true'][0]
    y = generated_normal_data['y']

    # True predictions will have log probability -> 1
    inds = np.where(np.sign(np.dot(X, weights)) == y)[0]
    assert np.all(log_loss.log_probs(Z, weights)[inds] > .99)
