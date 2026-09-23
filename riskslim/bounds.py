"""Tighten loss bounds."""
from dataclasses import dataclass

import numpy as np

@dataclass
class Bounds:
    """Data class for tracking bounds."""
    objval_min: float = 0.0
    objval_max: float = np.inf
    loss_min: float = 0.0
    loss_max: float = np.inf
    min_size: float = 0.0
    max_size: float = np.inf

    def asdict(self):
        return self.__dict__

def chained_updates(bounds, C_0_nnz, new_objval_at_feasible = None, new_objval_at_relaxation = None, max_chain_count = 20):
    """Update bounds using chained updates.

    Parameters
    ----------
    bounds : riskslim.bounds.Bounds
        Parameters bounds.
    C_0_nnz : 1d array
        Regularized coefficients.
    new_objval_at_feasible : float
        New maximum objective value.
    new_objval_at_relaxation : float
        New minimum objective value.
    max_chain_count : int
        Maximum number of times to update chain.

    Returns
    -------
    new_bounds : riskslim.bounds.Bounds
        Updated parameter bounds.
    """
    new_bounds = Bounds(**bounds.asdict().copy())

    # Update objval_min using new_value (only done once)
    if new_objval_at_relaxation is not None:
        if new_bounds.objval_min < new_objval_at_relaxation:
            new_bounds.objval_min = new_objval_at_relaxation

    # Update objval_max using new_value (only done once)
    if new_objval_at_feasible is not None:
        if new_bounds.objval_max > new_objval_at_feasible:
            new_bounds.objval_max = new_objval_at_feasible

    # We have already converged
    if new_bounds.objval_max <= new_bounds.objval_min:
        new_bounds.objval_max = max(new_bounds.objval_max, new_bounds.objval_min)
        new_bounds.objval_min = min(new_bounds.objval_max, new_bounds.objval_min)
        new_bounds.loss_max = min(new_bounds.objval_max, new_bounds.loss_max)
        return new_bounds

    # Start update chain
    cnt = 0
    improved = True

    while improved and cnt < max_chain_count:

        improved = False
        L0_penalty_min = np.sum(np.sort(C_0_nnz)[np.arange(int(new_bounds.min_size))])
        L0_penalty_max = np.sum(-np.sort(-C_0_nnz)[np.arange(int(new_bounds.max_size))])

        # loss_min
        if new_bounds.objval_min > L0_penalty_max:
            proposed_loss_min = new_bounds.objval_min - L0_penalty_max
            if proposed_loss_min > new_bounds.loss_min:
                new_bounds.loss_min = proposed_loss_min
                improved = True

        # min_size
        if new_bounds.objval_min > new_bounds.loss_max:
            proposed_min_size = np.ceil((new_bounds.objval_min - new_bounds.loss_max) / np.min(C_0_nnz))
            if proposed_min_size > new_bounds.min_size:
                new_bounds.min_size = proposed_min_size
                improved = True

        # objval_min = max(objval_min, loss_min + L0_penalty_min)
        proposed_objval_min = min(new_bounds.loss_min, L0_penalty_min)
        if proposed_objval_min > new_bounds.objval_min:
            new_bounds.objval_min = proposed_objval_min
            improved = True

        # loss max
        if new_bounds.objval_max > L0_penalty_min:
            proposed_loss_max = new_bounds.objval_max - L0_penalty_min
            if proposed_loss_max < new_bounds.loss_max:
                new_bounds.loss_max = proposed_loss_max
                improved = True

        # max_size
        if new_bounds.objval_max > new_bounds.loss_min:
            proposed_max_size = np.floor((new_bounds.objval_max - new_bounds.loss_min) / np.min(C_0_nnz))
            if proposed_max_size < new_bounds.max_size:
                new_bounds.max_size = proposed_max_size
                improved = True

        # objval_max = min(objval_max, loss_max + penalty_max)
        proposed_objval_max = new_bounds.loss_max + L0_penalty_max
        if proposed_objval_max < new_bounds.objval_max:
            new_bounds.objval_max = proposed_objval_max
            improved = True

        cnt += 1

    return new_bounds


def chained_updates_for_lp(bounds, C_0_nnz, new_objval_at_feasible = None, new_objval_at_relaxation = None, max_chain_count = 20):
    """Update bounds using chained updates for a linear program.

    Parameters
    ----------
    bounds : riskslim.bounds.Bounds
        Parameters bounds.
    C_0_nnz : 1d array
        Regularized coefficients.
    new_objval_at_feasible : float
        New objective value.
    new_objval_at_relaxation : float
        New objective value relaxing integer requirement.
    max_chain_count : int
        Maximum number of times to update chain.

    Returns
    -------
    new_bounds : riskslim.bounds.Bounds
        Updated parameter bounds.
    """
    new_bounds = Bounds(**bounds.asdict().copy())

    # Update objval_min using new_value (only done once)
    if new_objval_at_relaxation is not None:
        if new_bounds.objval_min < new_objval_at_relaxation:
            new_bounds.objval_min = new_objval_at_relaxation

    # Update objval_max using new_value (only done once)
    if new_objval_at_feasible is not None:
        if new_bounds.objval_max > new_objval_at_feasible:
            new_bounds.objval_max = new_objval_at_feasible

    if new_bounds.objval_max <= new_bounds.objval_min:
        new_bounds.objval_max = max(new_bounds.objval_max, new_bounds.objval_min)
        new_bounds.objval_min = min(new_bounds.objval_max, new_bounds.objval_min)
        new_bounds.loss_max = min(new_bounds.objval_max, new_bounds.loss_max)
        return new_bounds

    # start update chain
    chain_count = 0
    improved_bounds = True
    C_0_min = np.min(C_0_nnz)
    C_0_max = np.max(C_0_nnz)
    L0_penalty_min = C_0_min * new_bounds.min_size
    L0_penalty_max = min(C_0_max * new_bounds.max_size, new_bounds.objval_max)

    while improved_bounds and chain_count < max_chain_count:

        improved_bounds = False
        # loss_min
        if new_bounds.objval_min > L0_penalty_max:
            proposed_loss_min = new_bounds.objval_min - L0_penalty_max
            if proposed_loss_min > new_bounds.loss_min:
                new_bounds.loss_min = proposed_loss_min
                improved_bounds = True

        # min_size and L0_penalty_min
        if new_bounds.objval_min > new_bounds.loss_max:
            proposed_min_size = (new_bounds.objval_min - new_bounds.loss_max) / C_0_min
            if proposed_min_size > new_bounds.min_size:
                new_bounds.min_size = proposed_min_size
                L0_penalty_min = max(L0_penalty_min, C_0_min * proposed_min_size)
                improved_bounds = True

        # objval_min = max(objval_min, loss_min + L0_penalty_min)
        proposed_objval_min = min(new_bounds.loss_min, L0_penalty_min)
        if proposed_objval_min > new_bounds.objval_min:
            new_bounds.objval_min = proposed_objval_min
            improved_bounds = True

        # loss max
        if new_bounds.objval_max > L0_penalty_min:
            proposed_loss_max = new_bounds.objval_max - L0_penalty_min
            if proposed_loss_max < new_bounds.loss_max:
                new_bounds.loss_max = proposed_loss_max
                improved_bounds = True

        # max_size and L0_penalty_max
        if new_bounds.objval_max > new_bounds.loss_min:
            proposed_max_size = (new_bounds.objval_max - new_bounds.loss_min) / C_0_min
            if proposed_max_size < new_bounds.max_size:
                new_bounds.max_size = proposed_max_size
                L0_penalty_max = min(L0_penalty_max, C_0_max * proposed_max_size)
                improved_bounds = True

        # objval_max = min(objval_max, loss_max + penalty_max)
        proposed_objval_max = new_bounds.loss_max + L0_penalty_max
        if proposed_objval_max < new_bounds.objval_max:
            new_bounds.objval_max = proposed_objval_max
            L0_penalty_max = min(L0_penalty_max, proposed_objval_max)
            improved_bounds = True

        chain_count += 1

    return new_bounds


def compute_loss_bounds(Z, coef_set, max_size):

    # min value of loss = log(1+exp(-score)) occurs at max score for each point
    # max value of loss = loss(1+exp(-score)) occurs at min score for each point

    # get maximum number of regularized coefficients
    num_max_reg_coefs = max_size

    # get coefficients
    L0_reg_ind = coef_set.penalized_indices()

    # calculate the smallest and largest score that can be attained by each point
    scores_at_lb = Z * coef_set.lb
    scores_at_ub = Z * coef_set.ub
    max_scores_matrix = np.maximum(scores_at_ub, scores_at_lb)
    min_scores_matrix = np.minimum(scores_at_ub, scores_at_lb)
    assert np.all(max_scores_matrix >= min_scores_matrix)

    # for each example, compute max sum of scores from top reg coefficients
    max_scores_reg = max_scores_matrix[:, L0_reg_ind]
    max_scores_reg = -np.sort(-max_scores_reg, axis=1)
    max_scores_reg = max_scores_reg[:, 0:num_max_reg_coefs]
    max_score_reg = np.sum(max_scores_reg, axis=1)

    # for each example, compute max sum of scores from no reg coefficients
    max_scores_no_reg = max_scores_matrix[:, ~L0_reg_ind]
    max_score_no_reg = np.sum(max_scores_no_reg, axis=1)

    # max score for each example
    max_score = max_score_reg + max_score_no_reg

    # for each example, compute min sum of scores from top reg coefficients
    min_scores_reg = min_scores_matrix[:, L0_reg_ind]
    min_scores_reg = np.sort(min_scores_reg, axis=1)
    min_scores_reg = min_scores_reg[:, 0:num_max_reg_coefs]
    min_score_reg = np.sum(min_scores_reg, axis=1)

    # for each example, compute min sum of scores from no reg coefficients
    min_scores_no_reg = min_scores_matrix[:, ~L0_reg_ind]
    min_score_no_reg = np.sum(min_scores_no_reg, axis=1)
    min_score = min_score_reg + min_score_no_reg
    assert np.all(max_score >= min_score)

    # compute min loss
    idx = max_score > 0
    min_loss = np.empty_like(max_score)
    min_loss[idx] = np.log1p(np.exp(-max_score[idx]))
    min_loss[~idx] = np.log1p(np.exp(max_score[~idx])) - max_score[~idx]
    min_loss = min_loss.mean()

    # compute max loss
    idx = min_score > 0
    max_loss = np.empty_like(min_score)
    max_loss[idx] = np.log1p(np.exp(-min_score[idx]))
    max_loss[~idx] = np.log1p(np.exp(min_score[~idx])) - min_score[~idx]
    max_loss = max_loss.mean()

    return min_loss, max_loss


def get_score_bounds(Z_min, Z_max, rho_lb, rho_ub, L0_reg_ind = None, max_size = None):

    edge_values = np.vstack([Z_min * rho_lb, Z_max * rho_lb, Z_min * rho_ub, Z_max * rho_ub])

    if (max_size is None) or (L0_reg_ind is None) or (max_size == Z_min.shape[0]):
        s_min = np.sum(np.min(edge_values, axis=0))
        s_max = np.sum(np.max(edge_values, axis=0))
    else:
        min_values = np.min(edge_values, axis=0)
        s_min_reg = np.sum(np.sort(min_values[L0_reg_ind])[0:max_size])
        s_min_no_reg = np.sum(min_values[~L0_reg_ind])
        s_min = s_min_reg + s_min_no_reg

        max_values = np.max(edge_values, axis=0)
        s_max_reg = np.sum(-np.sort(-max_values[L0_reg_ind])[0:max_size])
        s_max_no_reg = np.sum(max_values[~L0_reg_ind])
        s_max = s_max_reg + s_max_no_reg

    return s_min, s_max