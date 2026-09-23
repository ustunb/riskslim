import numpy as np


def sequential_rounding(rho, Z, C_0, compute_loss_from_scores, get_L0_penalty, objval_cutoff = float('Inf')):
    """

    Parameters
    ----------
    rho:                                P x 1 vector of continuous coefficients
    Z:                                  N x P data matrix computed as X * Y
    C_0:                                N x 1 vector of L0 penalties. C_0[j] = L0 penalty for rho[j] for j = 0,..., P.
    compute_loss_from_scores:      function handle to compute loss using N x 1 vector of scores, where scores = Z.dot(rho)
    get_L0_penalty:                     function handle to compute L0_penalty from rho
    objval_cutoff:                      objective value used for early stopping.
                                        the procedure will stop if the objective value achieved by an intermediate solution will exceeds objval_cutoff

    Returns
    -------

    rho:                                P x 1 vector of integer coefficients (if early_stop_flag = False, otherwise continuous solution)
    best_objval:                        objective value achieved by rho (if early_stop_flag = False, otherwise NaN)
    early_stop_flag:                    True if procedure was stopped early (in which case rho is not integer feasible)

    """

    assert callable(compute_loss_from_scores)
    assert callable(get_L0_penalty)

    rho = np.copy(rho)
    d = rho.shape[0]

    rho_floor = np.floor(rho)
    floor_is_zero = np.equal(rho_floor, 0)
    dist_from_start_to_floor = rho_floor - rho

    rho_ceil = np.ceil(rho)
    ceil_is_zero = np.equal(rho_ceil, 0)
    dist_from_start_to_ceil = rho_ceil - rho

    dimensions_to_round = np.flatnonzero(np.not_equal(rho_floor, rho_ceil)).tolist()

    scores = Z.dot(rho)
    best_objval = compute_loss_from_scores(scores) + get_L0_penalty(rho)
    while len(dimensions_to_round) > 0 and best_objval < objval_cutoff:

        objvals_at_floor = np.repeat(np.nan, d)
        objvals_at_ceil = np.repeat(np.nan, d)
        current_penalty = get_L0_penalty(rho)

        for idx in dimensions_to_round:

            # scores go from center to ceil -> center + dist_from_start_to_ceil
            Z_dim = Z[:, idx]
            base_scores = scores + dist_from_start_to_ceil[idx] * Z_dim
            objvals_at_ceil[idx] = compute_loss_from_scores(base_scores)

            # move from ceil to floor => -1*Z_j
            base_scores -= Z_dim
            objvals_at_floor[idx] = compute_loss_from_scores(base_scores)

            if ceil_is_zero[idx]:
                objvals_at_ceil[idx] -= C_0[idx]
            elif floor_is_zero[idx]:
                objvals_at_floor[idx] -= C_0[idx]


        # adjust for penalty value
        objvals_at_ceil += current_penalty
        objvals_at_floor += current_penalty
        best_objval_at_ceil = np.nanmin(objvals_at_ceil)
        best_objval_at_floor = np.nanmin(objvals_at_floor)

        if best_objval_at_ceil <= best_objval_at_floor:
            best_objval = best_objval_at_ceil
            best_dim = np.nanargmin(objvals_at_ceil)
            rho[best_dim] += dist_from_start_to_ceil[best_dim]
            scores += dist_from_start_to_ceil[best_dim] * Z[:, best_dim]
        else:
            best_objval = best_objval_at_floor
            best_dim = np.nanargmin(objvals_at_floor)
            rho[best_dim] += dist_from_start_to_floor[best_dim]
            scores += dist_from_start_to_floor[best_dim] * Z[:, best_dim]

        dimensions_to_round.remove(best_dim)
        #assert(np.all(np.isclose(scores, Z.dot(rho))))

    early_stop_flag = best_objval > objval_cutoff
    return rho, best_objval, early_stop_flag


def discrete_descent(rho, Z, C_0, rho_ub, rho_lb, get_L0_penalty, compute_loss_from_scores, descent_dimensions = None, active_set_flag = True):

    """
    Given a initial feasible solution, rho, produces an improved solution that is 1-OPT
    (i.e. the objective value does not decrease by moving in any single dimension)
    at each iteration, the algorithm moves in the dimension that yields the greatest decrease in objective value
    the best step size is each dimension is computed using a directional search strategy that saves computation

    Parameters
    ----------
    rho:                                P x 1 vector of continuous coefficients
    Z:                                  N x P data matrix computed as X * Y
    C_0:                                N x 1 vector of L0 penalties. C_0[j] = L0 penalty for rho[j] for j = 0,..., P.
    rho_ub
    rho_lb
    compute_loss_from_scores:      function handle to compute loss using N x 1 vector of scores, where scores = Z.dot(rho)
    get_L0_penalty:                     function handle to compute L0_penalty from rho
    descent_dimensions

    Returns
    -------

    """

    assert callable(compute_loss_from_scores)
    assert callable(get_L0_penalty)

    # initialize key variables
    MAX_ITERATIONS = 500
    MIN_IMPROVEMENT_PER_STEP = float(1e-8)
    d = len(rho)

    # convert solution to integer
    rho = np.require(np.require(rho, dtype = np.int_), dtype = np.float64)

    # convert descent dimensions to integer values
    if descent_dimensions is None:
        descent_dimensions = np.arange(d)
    else:
        descent_dimensions = np.require(descent_dimensions, dtype = np.int_)

    if active_set_flag:
        descent_dimensions = np.intersect1d(np.flatnonzero(rho), descent_dimensions)

    descent_dimensions = descent_dimensions.tolist()

    base_scores = Z.dot(rho)
    base_loss = compute_loss_from_scores(base_scores)
    base_objval = base_loss + get_L0_penalty(rho)
    n_iterations = 0

    coefficient_values = {k: np.arange(int(rho_lb[k]), int(rho_ub[k]) + 1) for k in descent_dimensions}
    search_dimensions = descent_dimensions
    while n_iterations < MAX_ITERATIONS and len(search_dimensions) > 0:

        # compute the best objective value / step size in each dimension
        best_objval_by_dim = np.repeat(np.nan, d)
        best_coef_by_dim = np.repeat(np.nan, d)

        for k in search_dimensions:

            dim_objvals = _compute_objvals_at_dim(base_rho = rho,
                                                  base_scores = base_scores,
                                                  base_loss = base_loss,
                                                  dim_idx = k,
                                                  dim_coefs = coefficient_values[k],
                                                  Z = Z,
                                                  C_0 = C_0,
                                                  compute_loss_from_scores = compute_loss_from_scores)

            # mark points that will improve the current objective value by at least MIN_IMPROVEMENT_PER_STEP
            best_dim_idx = np.nanargmin(dim_objvals)
            best_objval_by_dim[k] = dim_objvals[best_dim_idx]
            best_coef_by_dim[k] = coefficient_values[k][best_dim_idx]

        # recompute base objective value/loss/scores
        best_idx = np.nanargmin(best_objval_by_dim)
        next_objval = best_objval_by_dim[best_idx]
        threshold_objval = base_objval - MIN_IMPROVEMENT_PER_STEP

        if next_objval >= threshold_objval:
            break

        best_step = best_coef_by_dim[best_idx] - rho[best_idx]
        rho[best_idx] += best_step
        base_objval = next_objval
        base_loss = base_objval - get_L0_penalty(rho)
        base_scores = base_scores + (best_step * Z[:, best_idx])

        # remove the current best direction from the set of directions to explore
        search_dimensions = list(descent_dimensions)
        search_dimensions.remove(best_idx)
        n_iterations += 1

    return rho, base_loss, base_objval


def _compute_objvals_at_dim(Z, C_0, base_rho, base_scores, base_loss, dim_coefs, dim_idx, compute_loss_from_scores):

    """
    finds the value of rho[j] in dim_coefs that minimizes log_loss(rho) + C_0j

    Parameters
    ----------
    Z
    C_0
    base_rho
    base_scores
    base_loss
    dim_coefs
    dim_idx
    compute_loss_from_scores

    Returns
    -------

    """

    # copy stuff because ctypes
    scores = np.copy(base_scores)

    # initialize parameters
    P = base_rho.shape[0]
    base_coef_value = base_rho[dim_idx]
    base_index = np.flatnonzero(dim_coefs == base_coef_value)
    loss_at_coef_value = np.repeat(np.nan, len(dim_coefs))
    loss_at_coef_value[base_index] = float(base_loss)
    Z_dim = Z[:, dim_idx]

    # start by moving forward
    forward_indices = np.flatnonzero(base_coef_value <= dim_coefs)
    forward_step_sizes = np.diff(dim_coefs[forward_indices] - base_coef_value)
    n_forward_steps = len(forward_step_sizes)
    stop_after_first_forward_step = False

    best_loss = base_loss
    total_distance_from_base = 0

    for i in range(n_forward_steps):
        scores += forward_step_sizes[i] * Z_dim
        total_distance_from_base += forward_step_sizes[i]
        current_loss = compute_loss_from_scores(scores)
        if current_loss >= best_loss:
            stop_after_first_forward_step = i == 0
            break
        loss_at_coef_value[forward_indices[i + 1]] = current_loss
        best_loss = current_loss

    # if the first step forward didn't lead to a decrease in loss, then move backwards
    move_backward = stop_after_first_forward_step or n_forward_steps == 0

    if move_backward:

        # compute backward steps
        backward_indices = np.flipud(np.where(dim_coefs <= base_coef_value)[0])
        backward_step_sizes = np.diff(dim_coefs[backward_indices] - base_coef_value)
        n_backward_steps = len(backward_step_sizes)

        # correct size of first backward step if you took 1 step forward
        if n_backward_steps > 0 and n_forward_steps > 0:
            backward_step_sizes[0] = backward_step_sizes[0] - forward_step_sizes[0]

        best_loss = base_loss

        for i in range(n_backward_steps):
            scores += backward_step_sizes[i] * Z_dim
            total_distance_from_base += backward_step_sizes[i]
            current_loss = compute_loss_from_scores(scores)
            if current_loss >= best_loss:
                break
            loss_at_coef_value[backward_indices[i + 1]] = current_loss
            best_loss = current_loss

    # at this point scores == base_scores + step_distance*Z_dim
    # assert(all(np.isclose(scores, base_scores + total_distance_from_base * Z_dim)))

    # compute objective values by adding penalty values to all other indices
    other_dim_idx = np.flatnonzero(dim_idx != np.arange(P))
    other_dim_penalty = np.sum(C_0[other_dim_idx] * (base_rho[other_dim_idx] != 0))
    objval_at_coef_values = loss_at_coef_value + other_dim_penalty

    if C_0[dim_idx] > 0.0:

        # increase objective value at every non-zero coefficient value by C_0j
        nonzero_coef_idx = np.flatnonzero(dim_coefs)
        objval_at_coef_values[nonzero_coef_idx] = objval_at_coef_values[nonzero_coef_idx] + C_0[dim_idx]

        # compute value at coef[j] == 0 if needed
        zero_coef_idx = np.flatnonzero(dim_coefs == 0)
        if np.isnan(objval_at_coef_values[zero_coef_idx]):
            # steps_from_here_to_zero: step_from_here_to_base + step_from_base_to_zero
            # steps_from_here_to_zero: -step_from_base_to_here + -step_from_zero_to_base
            steps_to_zero = -(base_coef_value + total_distance_from_base)
            scores += steps_to_zero * Z_dim
            objval_at_coef_values[zero_coef_idx] = compute_loss_from_scores(scores) + other_dim_penalty
            # assert(all(np.isclose(scores, base_scores - base_coef_value * Z_dim)))

    # return objective value at feasible coefficients
    return objval_at_coef_values

