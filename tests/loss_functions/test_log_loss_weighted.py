"""Tests for weighted log loss."""

import numpy as np
from riskslim.loss_functions import log_loss, log_loss_weighted


def test_log_loss_value(generated_normal_data):
    """Test weighted log loss value."""

    # Unpack fixture
    Z = generated_normal_data['Z'][0]
    rho = generated_normal_data['rho']

    # Weights of one are the same as unweighted
    assert log_loss_weighted.log_loss_value(Z, np.ones(len(Z)), len(Z), rho).round(6) == \
        log_loss.log_loss_value(Z, rho).round(6)

    # Weights of zero should give zero loss
    assert log_loss_weighted.log_loss_value(Z, np.zeros(len(Z)), len(Z), rho) == 0.

    # Loss of random row in Z, selected using weights
    ind = np.random.choice(np.arange(len(Z)))
    weights = np.zeros(len(Z))
    weights[ind] = 1

    assert log_loss_weighted.log_loss_value(Z, weights, len(Z), rho) == \
        log_loss.log_loss_value(Z[ind], rho) / len(Z)


def test_log_loss_value_and_slope(generated_normal_data):
    """Test weighted log loss value and slope."""

    # Unpack fixture
    Z = generated_normal_data['Z'][0]
    rho = generated_normal_data['rho']

    # Weights are all one, assert equal to unweighted logloss
    wloss, wslope = log_loss_weighted.log_loss_value_and_slope(
        Z, np.ones(len(Z)), len(Z), rho)

    loss, slope = log_loss.log_loss_value_and_slope(Z, rho)

    assert wloss.round(6) == loss.round(6)
    assert np.all(slope == wslope)

    # Weights are all zero, assert zero
    wloss, wslope = log_loss_weighted.log_loss_value_and_slope(
        Z, np.zeros(len(Z)), len(Z), rho)

    assert wloss == 0
    assert np.all(wslope == 0)

    # Loss of random row in Z, selected using weights
    ind = np.random.choice(np.arange(len(Z)))
    weights = np.zeros(len(Z))
    weights[ind] = 1

    wloss, wslope = log_loss_weighted.log_loss_value_and_slope(Z, weights, len(Z), rho)
    loss, slope = log_loss.log_loss_value_and_slope(Z[ind], rho)

    # Float precision inaccuracy, use allclose to pass equality check
    np.testing.assert_allclose(slope * len(Z[0]) /(len(Z)), wslope)


def test_log_loss_value_from_scores(generated_normal_data):
    """Test weighted log loss from scores."""

    # Unpack fixture
    Z = generated_normal_data['Z'][0]
    rho = generated_normal_data['rho']

    # Weights are all one, assert equal to unweighted logloss
    wloss = log_loss_weighted.log_loss_value_from_scores(
        np.ones(len(Z)), len(Z), Z.dot(rho))

    loss = log_loss.log_loss_value_from_scores(Z.dot(rho))

    assert wloss.round(6)  == loss.round(6)

    # Loss of random row in Z, selected using weights
    ind = np.random.choice(np.arange(len(Z)))
    weights = np.zeros(len(Z))
    weights[ind] = 1

    wloss = log_loss_weighted.log_loss_value_from_scores(
        weights, len(Z), Z.dot(rho))

    loss = log_loss.log_loss_value_from_scores(Z[ind].dot(rho)) / len(Z)

    assert wloss == loss
