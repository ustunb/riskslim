import numpy as np
from cplex import Cplex, SparsePair, infinity as CPX_INFINITY
from .coefficient_set import CoefficientSet
from .defaults import DEFAULT_CPLEX_SETTINGS
from .utils import print_log

#todo: add loss cut
#todo: add constraint function
#todo: default cplex parameters
#todo: check cores
#todo: pass compute_loss to convert_risk_slim_cplex_solution

def create_risk_slim(coef_set, settings):
    """Create a RiskSLIM MIP object.

    Parameters
    ----------
    coef_set : riskslim.coefficient_set.CoefficientSet
        Contraints (bounds) on coefficients of settings variables.
    seetings : dict
        MIP settings.

    Returns
    -------
    mip : cplex.Cplex
        RiskSLIM surrogate MIP without 0 cuts.
    indices : dict
        Parameter indices of RiskSLIM surrogate MIP.

    Issues
    ------
    no support for non-integer Lset "values"
    only drops intercept index for variable_names that match '(Intercept)'

    Notes
    -----
    RiskSLIM MIP Formulation

    minimize w_pos*loss_pos + w_neg *loss_minus + 0*rho_j + C_0j*alpha_j

    such that

    min_size ≤ L0 ≤ max_size
    -rho_min * alpha_j < lambda_j < rho_max * alpha_j

    L_0 in 0 to P
    rho_j in [rho_min_j, rho_max_j]
    alpha_j in {0,1}

    x = [loss_pos, loss_neg, rho_j, alpha_j]

    Optional constraints:

    objval = w_pos * loss_pos + w_neg * loss_min + sum(C_0j * alpha_j) (required for callback)
    L0_norm = sum(alpha_j) (required for callback)

    Changes for Tight Formulation (included when settings['tight_formulation'] = True):

    sigma_j in {0,1} for j s.t. lambda_j has free sign and alpha_j exists
    lambda_j ≥ delta_pos_j if alpha_j = 1 and sigma_j = 1
    lambda_j ≥ -delta_neg_j if alpha_j = 1 and sigma_j = 0
    lambda_j ≥ alpha_j for j such that lambda_j >= 0
    lambda_j ≤ -alpha_j for j such that lambda_j <= 0
    """
    assert isinstance(coef_set, CoefficientSet)
    assert isinstance(settings, dict)

    # setup printing and loading
    function_print_flag = settings.get('print_flag', False)
    print_from_function = lambda msg: print_log(msg) if function_print_flag else lambda msg: None

    # set default parameters
    settings.setdefault('C_0', 0.01)
    settings.setdefault('w_pos', 1.0)
    settings.setdefault('w_neg', 2.0 - settings['w_pos'])
    settings.setdefault('include_auxillary_variable_for_objval', True)
    settings.setdefault('include_auxillary_variable_for_L0_norm', True)
    settings.setdefault('loss_min', 0.00)
    settings.setdefault('loss_max', float(CPX_INFINITY))
    settings.setdefault('min_size', 0)
    settings.setdefault('max_size', len(coef_set))
    settings.setdefault('objval_min', 0.00)
    settings.setdefault('objval_max', float(CPX_INFINITY))
    settings.setdefault('relax_integer_variables', False)
    settings.setdefault('drop_variables', True)
    settings.setdefault('tight_formulation', False)
    settings.setdefault('set_cplex_cutoffs', True)

    # variables
    P = len(coef_set)
    w_pos, w_neg = settings['w_pos'], settings['w_neg']
    C_0j = np.copy(coef_set.c0)
    L0_reg_ind = np.isnan(C_0j) + C_0j != 0.0
    C_0j[L0_reg_ind] = settings['C_0'][L0_reg_ind] if isinstance(settings['C_0'], np.ndarray) \
        else settings['C_0']
    C_0j = C_0j.tolist()
    C_0_rho = np.copy(C_0j)
    trivial_min_size = 0
    trivial_max_size = np.sum(L0_reg_ind)

    rho_ub = list(coef_set.ub)
    rho_lb = list(coef_set.lb)
    rho_type = ''.join(list(coef_set.vtype))

    # calculate min/max values for loss
    loss_min = max(0.0, float(settings['loss_min']))
    loss_max = min(CPX_INFINITY, float(settings['loss_max']))

    # calculate min/max values for model size
    min_size = np.maximum(settings['min_size'], 0.0)
    max_size = np.minimum(settings['max_size'], trivial_max_size)
    min_size = np.ceil(min_size)
    max_size = np.floor(max_size)
    assert min_size <= max_size

    # calculate min/max values for objval
    objval_min = max(settings['objval_min'], 0.0)
    objval_max = min(settings['objval_max'], CPX_INFINITY)
    assert objval_min <= objval_max

    # include constraint on min/max model size?
    nontrivial_min_size = min_size > trivial_min_size
    nontrivial_max_size = max_size < trivial_max_size
    include_auxillary_variable_for_L0_norm = settings['include_auxillary_variable_for_L0_norm'] or \
                                             nontrivial_min_size or \
                                             nontrivial_max_size

    # include constraint on min/max objective value?
    nontrivial_objval_min = objval_min > 0.0
    nontrivial_objval_max = objval_max < CPX_INFINITY
    include_auxillary_variable_for_objval = settings['include_auxillary_variable_for_objval'] or \
                                            nontrivial_objval_min or \
                                            nontrivial_objval_max

    has_intercept = '(Intercept)' in coef_set.variable_names

    # create MIP object
    mip = Cplex()
    vars = mip.variables
    cons = mip.linear_constraints

    # set sense
    mip.objective.set_sense(mip.objective.sense.minimize)

    # add main variables
    loss_obj = [w_pos]
    loss_ub = [loss_max]
    loss_lb = [loss_min]
    loss_type = 'C'
    loss_names = ['loss']

    obj = loss_obj + [0.0] * P + C_0j
    ub = loss_ub + rho_ub + [1.0] * P
    lb = loss_lb + rho_lb + [0.0] * P
    ctype = loss_type + rho_type + 'B' * P

    rho_names = ['rho_%d' % j for j in range(P)]
    alpha_names = ['alpha_%d' % j for j in range(P)]
    varnames = loss_names + rho_names + alpha_names

    if include_auxillary_variable_for_objval:
        objval_auxillary_name = ['objval']
        objval_auxillary_ub = [objval_max]
        objval_auxillary_lb = [objval_min]
        objval_type = 'C'

        print_from_function("adding auxiliary variable for objval s.t. %1.4f <= objval <= %1.4f" % (objval_min, objval_max))
        obj += [0.0]
        ub += objval_auxillary_ub
        lb += objval_auxillary_lb
        varnames += objval_auxillary_name
        ctype += objval_type

    if include_auxillary_variable_for_L0_norm:
        L0_norm_auxillary_name = ['L0_norm']
        L0_norm_auxillary_ub = [float(max_size)]
        L0_norm_auxillary_lb = [float(min_size)]
        L0_norm_type = 'I'

        print_from_function("adding auxiliary variable for L0_norm s.t. %d <= L0_norm <= %d" % (min_size, max_size))
        obj += [0.0]
        ub += L0_norm_auxillary_ub
        lb += L0_norm_auxillary_lb
        varnames += L0_norm_auxillary_name
        ctype += L0_norm_type

    if settings['relax_integer_variables']:
        ctype = ctype.replace('I', 'C')
        ctype = ctype.replace('B', 'C')

    vars.add(obj = obj, lb = lb, ub = ub, types = ctype, names = varnames)

    # 0-Norm LB Constraints:
    # lambda_j,lb * alpha_j ≤ lambda_j <= Inf
    # 0 ≤ lambda_j - lambda_j,lb * alpha_j < Inf
    for j in range(P):
        cons.add(names = ["L0_norm_lb_" + str(j)],
                 lin_expr = [SparsePair(ind=[rho_names[j], alpha_names[j]], val=[1.0, -rho_lb[j]])],
                 senses = "G",
                 rhs = [0.0])

    # 0-Norm UB Constraints:
    # lambda_j ≤ lambda_j,ub * alpha_j
    # 0 <= -lambda_j + lambda_j,ub * alpha_j
    for j in range(P):
        cons.add(names = ["L0_norm_ub_" + str(j)],
                 lin_expr =[SparsePair(ind=[rho_names[j], alpha_names[j]], val=[-1.0, rho_ub[j]])],
                 senses = "G",
                 rhs = [0.0])

    # objval_max constraint
    # loss_var + sum(C_0j .* alpha_j) <= objval_max
    if include_auxillary_variable_for_objval:
        print_from_function("adding constraint so that objective value <= " + str(objval_max))
        cons.add(names = ["objval_def"],
                 lin_expr = [SparsePair(ind = objval_auxillary_name + loss_names + alpha_names, val=[-1.0] + loss_obj + C_0j)],
                 senses = "E",
                 rhs = [0.0])

    # Auxiliary L0_norm variable definition:
    # L0_norm = sum(alpha_j)
    # L0_norm - sum(alpha_j) = 0
    if include_auxillary_variable_for_L0_norm:
        cons.add(names = ["L0_norm_def"],
                 lin_expr = [SparsePair(ind = L0_norm_auxillary_name + alpha_names, val = [1.0] + [-1.0] * P)],
                 senses = "E",
                 rhs = [0.0])


    # drop L0_norm_lb constraint for any variable with rho_lb >= 0
    dropped_variables = []
    constraints_to_drop = []

    # drop alpha / L0_norm_ub / L0_norm_lb for ('Intercept')
    if settings['drop_variables']:
        # drop L0_norm_ub/lb constraint for any variable with rho_ub/rho_lb >= 0
        sign_pos_ind = np.flatnonzero(coef_set.sign > 0)
        sign_neg_ind = np.flatnonzero(coef_set.sign < 0)
        constraints_to_drop.extend(["L0_norm_lb_" + str(j) for j in sign_pos_ind])
        constraints_to_drop.extend(["L0_norm_ub_" + str(j) for j in sign_neg_ind])

        # drop alpha for any variable where rho_ub = rho_lb = 0
        fixed_value_ind = np.flatnonzero(coef_set.ub == coef_set.lb)
        variables_to_drop = ["alpha_" + str(j) for j in fixed_value_ind]
        vars.delete(variables_to_drop)
        dropped_variables += variables_to_drop
        alpha_names = [alpha_names[j] for j in range(P) if alpha_names[j] not in dropped_variables]

    if has_intercept:
        intercept_idx = coef_set.variable_names.index('(Intercept)')
        intercept_alpha_name = 'alpha_' + str(intercept_idx)

        # Intercept may be previously dropped above when rho_ub == rho_lb
        if intercept_alpha_name not in dropped_variables:
            vars.delete([intercept_alpha_name])
            alpha_names.remove(intercept_alpha_name)
            dropped_variables.append(intercept_alpha_name)

        print_from_function("dropped L0 indicator for '(Intercept)'")
        constraints_to_drop.extend(["L0_norm_ub_" + str(intercept_idx), "L0_norm_lb_" + str(intercept_idx)])

    if len(constraints_to_drop) > 0:
        constraints_to_drop = list(set(constraints_to_drop))
        cons.delete(constraints_to_drop)

    # indices
    indices = {
        'n_variables': vars.get_num(),
        'n_constraints': cons.get_num(),
        'names': vars.get_names(),
        'loss_names': loss_names,
        'rho_names': rho_names,
        'alpha_names': alpha_names,
        'loss': vars.get_indices(loss_names),
        'rho': vars.get_indices(rho_names),
        'alpha': vars.get_indices(alpha_names),
        'L0_reg_ind': L0_reg_ind,
        'C_0_rho': C_0_rho,
        'C_0_alpha': mip.objective.get_linear(alpha_names) if len(alpha_names) > 0 else [],
        }

    if include_auxillary_variable_for_objval:
        indices.update({
            'objval_name': objval_auxillary_name,
            'objval': vars.get_indices(objval_auxillary_name)[0],
            })

    if include_auxillary_variable_for_L0_norm:
        indices.update({
            'L0_norm_name': L0_norm_auxillary_name,
            'L0_norm': vars.get_indices(L0_norm_auxillary_name)[0],
            })

    # officially change the problem to LP if variables are relaxed
    if settings['relax_integer_variables']:
        old_problem_type = mip.problem_type[mip.get_problem_type()]
        mip.set_problem_type(mip.problem_type.LP)
        new_problem_type = mip.problem_type[mip.get_problem_type()]
        print_from_function("changed problem type from %s to %s" % (old_problem_type, new_problem_type))

    if settings['set_cplex_cutoffs'] and not settings['relax_integer_variables']:
        mip.parameters.mip.tolerances.lowercutoff.set(objval_min)
        mip.parameters.mip.tolerances.uppercutoff.set(objval_max)

    return mip, indices


def set_cplex_mip_parameters(mip, param, display_cplex_progress=False):
    """Helper function to set CPLEX parameters of CPLEX MIP object

    Parameters
    ----------
    mip : cplex.Cplex
        RiskSLIM surrogate MIP without 0 cuts.
    param : dict
        Parameter settings.
    display_cplex_progress : bool, optional, default: False
        Displays cplex progress when True, supresses if False.

    Returns
    -------
    mip : cplex.Cplex
        RiskSLIM surrogate MIP without 0 cuts.
    """
    p = mip.parameters
    p.randomseed.set(param['randomseed'])
    p.threads.set(param['n_cores'])
    p.output.clonelog.set(0)
    p.parallel.set(1)

    if display_cplex_progress is (None or False):
        mip = set_cpx_display_options(mip, display_mip = False, display_lp = False, display_parameters = False)

    p.simplex.tolerances.optimality.set(
        param.get('optimality_tolerance', DEFAULT_CPLEX_SETTINGS['optimality_tolerance'])
    )
    problem_type = mip.problem_type[mip.get_problem_type()]

    if problem_type == 'MILP':
        # CPLEX Memory Parameters
        # MIP.Param.workdir.Cur  = exp_workdir;
        # MIP.Param.workmem.Cur                    = cplex_workingmem;
        # MIP.Param.mip.strategy.file.Cur          = 2; %nodefile uncompressed
        # MIP.Param.mip.limits.treememory.Cur      = cplex_nodefilesize;

        # CPLEX MIP Parameters
        p.emphasis.mip.set(param['mipemphasis'])
        p.mip.tolerances.mipgap.set(param['mipgap'])
        p.mip.tolerances.absmipgap.set(param['absmipgap'])
        p.mip.tolerances.integrality.set(param['integrality_tolerance'])

        # CPLEX Solution Pool Parameters
        p.mip.limits.repairtries.set(param['repairtries'])
        p.mip.pool.capacity.set(param['poolsize'])
        p.mip.pool.replace.set(param['poolreplace'])
        # 0 = replace oldest /1: replace worst objective / #2 = replace least diverse solutions

    return mip


def set_cpx_display_options(mip, display_mip=True, display_parameters=False, display_lp=False):
    """Helper function to set display options of CPLEX MIP object.

    Parameters
    ----------
    mip : cplex.Cplex
        RiskSLIM surrogate MIP without 0 cuts.
    display_mip : bool, optional, default: True
        Displays MIP progress.
    display_parameters : bool, optional, default: False
        Displays parameter updates.
    display_lp : bool, optional, default: False
        Dsiplays linear program updates.

    Returns
    -------
    mip : cplex.Cplex
        RiskSLIM surrogate MIP without 0 cuts.
    """
    mip.parameters.mip.display.set(display_mip)
    mip.parameters.simplex.display.set(display_lp)

    try:
        mip.parameters.paramdisplay.set(display_parameters)
    except AttributeError:
        pass

    if not (display_mip or display_lp):
        mip.set_results_stream(None)
        mip.set_log_stream(None)
        mip.set_error_stream(None)
        mip.set_warning_stream(None)

    return mip


def add_mip_starts(mip, indices, pool, max_mip_starts=float('inf'), mip_start_effort_level=4):
    """Add initial solutions from pool to MIP object.

    Parameters
    ----------
    mip : cplex.Cplex
        RiskSLIM surrogate MIP without 0 cuts.
    indices : dict
        Parameter indices of RiskSLIM surrogate MIP.
    pool : riskslim.solution_pool.SolutionPool
        Pool of riskSLIM solutions.
    max_mip_starts : float, optional, default: inf
        Max number of mip starts to add. Default is add all.
    mip_start_effort_level : int
        Effort that CPLEX will spend trying to fix.

    Returns
    -------
    mip : cplex.Cplex
        RiskSLIM surrogate MIP without 0 cuts.
    """
    # todo remove suboptimal using pool filter
    assert isinstance(mip, Cplex)

    try:
        obj_cutoff = mip.parameters.mip.tolerances.uppercutoff.get()
    except:
        obj_cutoff = float('inf')

    pool = pool.distinct().sort()

    n_added = 0
    for objval, rho in zip(pool.objvals, pool.solutions):
        if np.less_equal(objval, obj_cutoff):
            mip_start_name = "mip_start_" + str(n_added)
            mip_start_obj, _ = convert_to_risk_slim_cplex_solution(rho = rho, indices = indices, objval = objval)
            mip_start_obj = cast_mip_start(mip_start_obj, mip)
            mip.MIP_starts.add(mip_start_obj, mip_start_effort_level, mip_start_name)
            n_added += 1

        if n_added >= max_mip_starts:
            break

    return mip


def cast_mip_start(mip_start, mip):
    """Casts the solution values and indices in a Cplex SparsePair.

    Parameters
    ----------
    mip_start : cplex.SparsePair
        Pairs of parameter indices and values.
    mip : cplex.Cplex
        RiskSLIM surrogate MIP without 0 cuts.

    Returns
    -------
    cplex.SparsePair
        Where the indices are integers and the values for each variable
        match the variable type specified in CPLEX Object.
    """

    assert isinstance(mip, Cplex)
    assert isinstance(mip_start, SparsePair)
    vals = list(mip_start.val)
    idx = np.array(list(mip_start.ind), dtype = int).tolist()
    types = mip.variables.get_types(idx)

    for j, t in enumerate(types):
        if t in ['B', 'I']:
            vals[j] = int(vals[j])
        elif t in ['C']:
            vals[j] = float(vals[j])

    return SparsePair(ind = idx, val = vals)


def convert_to_risk_slim_cplex_solution(rho, indices, loss = None, objval = None):
    """Convert coefficient vector 'rho' into a solution for RiskSLIM CPLEX MIP.

    Parameters
    ----------
    rho : 1d array
        Model coefficients or solutions.
    indices : dict
        Parameter indices of RiskSLIM surrogate MIP.
    loss : float, optional, default: None
        Loss using rho.
    objval : float, optional, default: None
        Objective value as loss minus L0 penalty.

    Returns
    -------
    solution_cpx : cplex.SparsePair
        Solution for RiskSLIM CPLEX MIP.
    objval : float
        Objective value as loss minus L0 penalty.
    """
    global compute_loss
    n_variables = indices['n_variables']
    solution_idx = np.arange(n_variables)
    solution_val = np.zeros(n_variables)

    # rho
    solution_val[indices['rho']] = rho

    # alpha
    alpha = np.zeros(len(indices['alpha']))
    alpha[np.flatnonzero(rho[indices['L0_reg_ind']])] = 1.0
    solution_val[indices['alpha']] = alpha
    L0_penalty = np.sum(indices['C_0_alpha'] * alpha)

    # add loss / objval
    need_loss = 'loss' in indices
    need_objective_val = 'objval' in indices
    need_L0_norm = 'L0_norm' in indices
    need_sigma = 'sigma_names' in indices

    # check that we have the right length
    # COMMENT THIS OUT FOR DEPLOYMENT
    # if need_sigma:
    #     pass
    # else:
    #     assert (indices['n_variables'] == (len(rho) + len(alpha) + need_loss + need_objective_val + need_L0_norm))

    if need_loss:
        if loss is None:
            if objval is None:
                loss = compute_loss(rho)
            else:
                loss = objval - L0_penalty

        solution_val[indices['loss']] = loss

    if need_objective_val:
        if objval is None:
            if loss is None:
                objval = compute_loss(rho) + L0_penalty
            else:
                objval = loss + L0_penalty

        solution_val[indices['objval']] = objval

    if need_L0_norm:
        solution_val[indices['L0_norm']] = np.sum(alpha)

    if need_sigma:
        rho_for_sigma = np.array([indices['rho'][int(s.strip('sigma_'))] for s in indices['sigma_names']])
        solution_val[indices['sigma']] = np.abs(solution_val[rho_for_sigma])

    solution_cpx = SparsePair(ind = solution_idx, val = solution_val.tolist())
    return solution_cpx, objval
