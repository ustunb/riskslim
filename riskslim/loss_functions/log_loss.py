import numpy as np

def log_loss_value(Z, weights):
    """Computes the value and slope of the logistic loss in a numerically stable way.

    Parameters
    ----------
    Z : 2d array
        Containing training data with shape = (n_rows, n_cols)
        Inner product of X and y.
    weights : 1d array
        Coefficients with shape = (n_cols,)

    Returns
    -------
    loss_value : float
        Loss as 1/n_rows * sum(log( 1 .+ exp(-Z*weights)).

    Notes
    -----
    See also: http://stackoverflow.com/questions/20085768/.
    """
    scores = Z.dot(weights)
    pos_idx = scores > 0
    loss_value = np.empty_like(scores)
    loss_value[pos_idx] = np.log1p(np.exp(-scores[pos_idx]))
    loss_value[~pos_idx] = -scores[~pos_idx] + np.log1p(np.exp(scores[~pos_idx]))
    loss_value = loss_value.mean()
    return loss_value

def log_loss_value_and_slope(Z, weights):
    """Computes the value and slope of the logistic loss in a numerically stable way
    this function should only be used when generating cuts in cutting-plane algorithms
    (computing both the value and the slope at the same time is slightly cheaper).

    Parameters
    ----------
    Z : 2d array
        Containing training data with shape = (n_rows, n_cols)
        Inner product of X and y.
    weights : 1d array
        Coefficients with shape = (n_cols,)

    Returns
    -------
    loss_value : float
        Loss as 1/n_rows * sum(log( 1 .+ exp(-Z*weights)).
    loss_slope : 1d array
        Gradient as (n_cols x 1) vector = 1/n_rows * sum(-Z*weights ./ (1+exp(-Z*weights)).

    Notes
    -----
    See also: http://stackoverflow.com/questions/20085768/
    """
    scores = Z.dot(weights)
    pos_idx = scores > 0
    exp_scores_pos = np.exp(-scores[pos_idx])
    exp_scores_neg = np.exp(scores[~pos_idx])

    #compute loss value
    loss_value = np.empty_like(scores)
    loss_value[pos_idx] = np.log1p(exp_scores_pos)
    loss_value[~pos_idx] = -scores[~pos_idx] + np.log1p(exp_scores_neg)
    loss_value = loss_value.mean()

    #compute loss slope
    log_probs = np.empty_like(scores)
    log_probs[pos_idx]  = 1.0 / (1.0 + exp_scores_pos)
    log_probs[~pos_idx] = exp_scores_neg / (1.0 + exp_scores_neg)
    loss_slope = Z.T.dot(log_probs - 1.0) / Z.shape[0]

    return loss_value, loss_slope

def log_loss_value_from_scores(scores):
    """Computes the logistic loss value from a vector of scores in a numerically stable way
    where scores = Z.dot(weights).

    Parameters
    ----------
    scores : 1d array
        Dot product between Z and weights.

    Returns
    -------
    loss_value : float
        Loss as 1/n_rows * sum(log( 1 .+ exp(-Z*weights)).

    Notes
    -----
    See also: http://stackoverflow.com/questions/20085768/

    This function is used for heuristics (discrete_descent, sequential_rounding).
    to save computation when running the heuristics, we store the scores and
    call this function to compute the loss directly from the scores
    this reduces the need to recompute the dot product.
    """

    pos_idx = scores > 0
    loss_value = np.empty_like(scores)
    loss_value[pos_idx] = np.log1p(np.exp(-scores[pos_idx]))
    loss_value[~pos_idx] = -scores[~pos_idx] + np.log1p(np.exp(scores[~pos_idx]))
    loss_value = loss_value.mean()
    return loss_value

def log_probs(Z, weights):
    """Compute the probabilities of the logistic loss function in a way that is numerically stable.

    Parameters
    ----------
    Z : 2d array
        Containing training data with shape = (n_rows, n_cols)
        Inner product of X and y.
    weights : 1d array
        Coefficients with shape = (n_cols,).

    Returns
    -------
    log_probs : 1d array
        Probabilities under the logit model.

    Notes
    -----
    See also: http://stackoverflow.com/questions/20085768/
    """

    scores = Z.dot(weights)
    pos_idx = scores > 0
    log_probs = np.empty_like(scores)
    log_probs[pos_idx]  = 1.0 / (1.0 + np.exp(-scores[pos_idx]))
    log_probs[~pos_idx] = np.exp(scores[~pos_idx]) / (1.0 + np.exp(scores[~pos_idx]))
    return log_probs
