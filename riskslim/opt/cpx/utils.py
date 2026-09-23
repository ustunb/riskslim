import numpy as np
import pandas as pd
from functools import reduce
from operator import iconcat, concat
from cplex import Cplex, SparsePair
from cplex.exceptions import CplexError


#### Stamping ####
concatenate = lambda m: reduce(iconcat, m, [])

is_geq_or_close = lambda a, b: np.logical_or(np.greater_equal(a, b), np.isclose(a, b)).astype(int)

is_leq_or_close = lambda a, b: np.logical_or(np.less_equal(a, b), np.isclose(a, b)).astype(int)


def add_variable(cpx, name, obj, ub, lb, vtype):

    assert isinstance(cpx, Cplex)
    if isinstance(name, str):
        name = [name]
    elif isinstance(name, np.ndarray):
        name = name.tolist()

    nvars = len(name)

    # convert inputs
    if nvars == 1:

        # convert to list
        name = name if isinstance(name, list) else [name]
        obj = [float(obj[0])] if isinstance(obj, list) else [float(obj)]
        ub = [float(ub[0])] if isinstance(ub, list) else [float(ub)]
        lb = [float(lb[0])] if isinstance(lb, list) else [float(lb)]
        vtype = vtype if isinstance(vtype, list) else [vtype]

    else:

        # convert to list
        if isinstance(vtype, np.ndarray):
            vtype = vtype.tolist()
        elif isinstance(vtype, str):
            if len(vtype) == 1:
                vtype = nvars * [vtype]
            elif len(vtype) == nvars:
                vtype = list(vtype)
            else:
                raise ValueError(f"invalid length: len(vtype) = {len(vtype)}. expected either 1 or {nvars}")

        if isinstance(obj, np.ndarray):
            obj = obj.astype(float).tolist()
        elif isinstance(obj, list):
            if len(obj) == nvars:
                obj = [float(v) for v in obj]
            elif len(obj) == 1:
                obj = nvars * [float(obj[0])]
            else:
                raise ValueError(f"invalid length: len(obj) = {len(obj)}. expected either 1 or {nvars}")
        else:
            obj = nvars * [float(obj)]

        if isinstance(ub, np.ndarray):
            ub = ub.astype(float).tolist()
        elif isinstance(ub, list):
            if len(ub) == nvars:
                ub = [float(v) for v in ub]
            elif len(ub) == 1:
                ub = nvars * [float(ub[0])]
            else:
                raise ValueError(f"invalid length: len(ub) = {len(ub)}. expected either 1 or {nvars}")
        else:
            ub = nvars * [float(ub)]

        if isinstance(lb, np.ndarray):
            lb = lb.astype(float).tolist()
        elif isinstance(lb, list):
            if len(lb) == nvars:
                lb = [float(v) for v in lb]
            elif len(lb) == 1:
                lb = nvars * [float(lb[0])]
            else:
                raise ValueError(f"invalid length: len(lb) = {len(lb)}. expected either 1 or {nvars}")
        else:
            lb = nvars * [float(lb)]

    # check that all components are lists
    assert all(isinstance(var, list) for var in [name, obj, ub, lb, vtype])

    # check components
    for n in range(nvars):
        assert isinstance(name[n], str)
        assert isinstance(obj[n], float)
        assert isinstance(ub[n], float)
        assert isinstance(lb[n], float)
        assert isinstance(vtype[n], str)

    if (vtype.count(vtype[0]) == len(vtype)) and vtype[0] == cpx.variables.type.binary:
        cpx.variables.add(names = name, obj = obj, types = vtype)
    else:
        cpx.variables.add(names = name, obj = obj, ub = ub, lb = lb, types = vtype)


def get_common_row_indices(A, B):
    """
    A and B need to only contain unique rows
    :param A:
    :param B:
    :return:
    """

    nrows, ncols = A.shape

    dtype = {
        'names': [f'f{i}' for i in range(ncols)],
        'formats': ncols * [A.dtype]
        }

    common_rows = np.intersect1d(A.view(dtype), B.view(dtype))
    if common_rows.shape[0] == 0:
        common_idx = np.empty((0, 2), dtype = np.dtype('int'))
    else:
        # common_rows = common_rows.view(A.dtype).reshape(-1, ncols)
        a_rows_idx = np.flatnonzero(np.isin(A.view(dtype), common_rows))
        b_rows_idx = np.flatnonzero(np.isin(B.view(dtype), common_rows))
        common_idx = np.column_stack((a_rows_idx, b_rows_idx))

    return common_idx


##### LP ####
CPX_LP_PARAMETERS = {
    #
    'display': False,
    # Set to True to show CPLEX progress in console
    #
    'n_cores': 1,
    # Number of CPU cores to use in B & B
    # May have to set n_cores = 1 in order to use certain control callbacks in CPLEX 12.7.0 and earlier
    #
    'randomseed': 2338,
    # This parameter sets the random seed differently for diversity of solutions.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/RandomSeed.html
    #
    }


def set_lp_parameters(cpx, param = CPX_LP_PARAMETERS):

    if param['display'] in (None, False):
        cpx.parameters.mip.display.set(False)
        cpx.parameters.simplex.display.set(False)
        cpx.parameters.paramdisplay.set(False)
        cpx.set_log_stream(None)
        cpx.set_error_stream(None)
        cpx.set_results_stream(None)
        cpx.set_warning_stream(None)


    # major ones
    cpx.parameters.randomseed.set(param['randomseed'])
    cpx.parameters.threads.set(param['n_cores'])
    cpx.parameters.output.clonelog.set(0)
    cpx.parameters.parallel.set(1)

    return cpx


##### MIP #####
CPX_MIP_PARAMETERS = {
    #
    'display_cplex_progress': True,
    #set to True to show CPLEX progress in console
    #
    'n_cores': 8,
    # Number of CPU cores to use in B & B
    # May have to set n_cores = 1 in order to use certain control callbacks in CPLEX 12.7.0 and earlier
    #
    'randomseed': 0,
    # This parameter sets the random seed differently for diversity of solutions.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/RandomSeed.html
    #
    'time_limit': 1e75,
    # runtime before stopping,
    #
    'node_limit': 9223372036800000000,
    # number of nodes to process before stopping,
    #
    'mipgap': np.finfo('float').eps,
    # Sets a relative tolerance on the gap between the best integer objective and the objective of the best node remaining.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpGap.html
    #
    'absmipgap': 0.9, #np.finfo('float').eps,
    # Sets an absolute tolerance on the gap between the best integer objective and the objective of the best node remaining.
    # When this difference falls below the value of this parameter, the mixed integer optimization is stopped.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpAGap.html
    #
    'objdifference': 0.9,
    # Used to update the cutoff each time a mixed integer solution is found. This value is subtracted from objective
    # value of the incumbent update, so that the solver ignore solutions that will not improve the incumbent by at
    # least this amount.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ObjDif.html#
    #
    'integrality_tolerance': 0.0,
    # specifies the amount by which an variable can differ from an integer and be considered integer feasible. 0 is OK
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpInt.html
    #
    'mipemphasis': 0,
    # Controls trade-offs between speed, feasibility, optimality, and moving bounds in MIP.
    # 0     =	Balance optimality and feasibility; default
    # 1	    =	Emphasize feasibility over optimality
    # 2	    =	Emphasize optimality over feasibility
    # 3 	=	Emphasize moving best bound
    # 4	    =	Emphasize finding hidden feasible solutions
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/MIPEmphasis.html
    #
    'bound_strengthening': -1,
    # Decides whether to apply bound strengthening in mixed integer programs (MIPs).
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/BndStrenInd.html
    # -1    = cpx chooses
    # 0     = no bound strengthening
    # 1     = bound strengthening
    #
    'cover_cuts': -1,
    # Decides whether or not cover cuts should be generated for the problem.
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/Covers.html
    # -1    = Do not generate cover cuts
    # 0	    = Automatic: let CPLEX choose
    # 1	    = Generate cover cuts moderately
    # 2	    = Generate cover cuts aggressively
    # 3     = Generate cover cuts very  aggressively
    #
    'zero_half_cuts': -1,
    # Decides whether or not to generate zero-half cuts for the problem. (set to off since these are not effective)
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ZeroHalfCuts.html
    # -1    = Do not generate MIR cuts
    # 0	    = Automatic: let CPLEX choose
    # 1	    = Generate MIR cuts moderately
    # 2	    = Generate MIR cuts aggressively
    #
    'mir_cuts': -1,
    # Decides whether or not to generate mixed-integer rounding cuts for the problem. (set to off since these are not effective)
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/MIRCuts.html
    # -1    = Do not generate zero-half cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate zero-half cuts moderately
    # 2	    = Generate zero-half cuts aggressively
    #
    'implied_bound_cuts': 0,
    # Decides whether or not to generate valid implied bound cuts for the problem.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ImplBdLocal.html
    # -1    = Do not generate locally valid implied bound cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate locally valid implied bound cuts moderately
    # 2	    = Generate locally valid implied bound cuts aggressively
    # 3	    = Generate locally valid implied bound cuts very aggressively
    #
    'locally_implied_bound_cuts': 3,
    # Decides whether or not to generate locally valid implied bound cuts for the problem.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ImplBdLocal.html
    # -1    = Do not generate locally valid implied bound cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate locally valid implied bound cuts moderately
    # 2	    = Generate locally valid implied bound cuts aggressively
    # 3	    = Generate locally valid implied bound cuts very aggressively
    #
    'scale_parameters': 1,
    # Decides how to scale the problem matrix.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ScaInd.html
    # 0     = equilibration scaling
    # 1     = aggressive scaling
    # -1    = no scaling
    #
    'numerical_emphasis': 0,
    # Emphasizes precision in numerically unstable or difficult problems.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/NumericalEmphasis.html
    # 0     = off
    # 1     = on
    #
    'poolsize': 100,
    # Limits the number of solutions kept in the solution pool
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolCapacity.html
    # number of feasible solutions to keep in solution pool
    #
    'poolrelgap': float('nan'),
    # Sets a relative tolerance on the objective value for the solutions in the solution pool.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolGap.html
    #
    'poolreplace': 2,
    # Designates the strategy for replacing a solution in the solution pool when the solution pool has reached its capacity.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolReplace.html
    # 0	= Replace the first solution (oldest) by the most recent solution; first in, first out; default
    # 1	= Replace the solution which has the worst objective
    # 2	= Replace solutions in order to build a set of diverse solutions
    #
    'repairtries': 20,
    # Limits the attempts to repair an infeasible MIP start.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/RepairTries.html
    # -1	None: do not try to repair
    #  0	Automatic: let CPLEX choose; default
    #  N	Number of attempts
    #
    'nodefilesize': (120 * 1024) / 1,
    # size of the node file (for large scale problems)
    # if the B & B can no longer fit in memory, then CPLEX stores the B & B in a node file
    }


# Solution Statistics
def has_solution(cpx):
    """returns true if mip has a feasible solution"""
    out = False
    try:
        cpx.solution.get_values()
        out = True
    except CplexError:
        pass
    return out


def get_mip_stats(cpx):
    """returns information associated with the current best solution for the mip"""

    info = {
        'status': 'no solution exists',
        'status_code': float('nan'),
        'has_solution': False,
        'has_mipstats': False,
        'iterations': 0,
        'nodes_processed': 0,
        'nodes_remaining': 0,
        'values': float('nan'),
        'objval': float('nan'),
        'upperbound': float('nan'),
        'lowerbound': float('nan'),
        'gap': float('nan'),
        }

    try:
        sol = cpx.solution
        info.update({'status': sol.get_status_string(),
                     'status_code': sol.get_status(),
                     'iterations': sol.progress.get_num_iterations(),
                     'nodes_processed': sol.progress.get_num_nodes_processed(),
                     'nodes_remaining': sol.progress.get_num_nodes_remaining()})
        info['has_mipstats'] = True
    except CplexError:
        pass

    try:
        sol = cpx.solution
        # get_cutoff() is the incumbent side and get_best_objective() the bound side of the gap;
        # the incumbent is an upper bound when minimizing and a lower bound when maximizing
        if cpx.objective.get_sense() == cpx.objective.sense.maximize:
            upperbound, lowerbound = sol.MIP.get_best_objective(), sol.MIP.get_cutoff()
        else:
            upperbound, lowerbound = sol.MIP.get_cutoff(), sol.MIP.get_best_objective()
        info.update({'values': np.array(sol.get_values()),
                     'objval': sol.get_objective_value(),
                     'upperbound': upperbound,
                     'lowerbound': lowerbound,
                     'gap': sol.MIP.get_mip_relative_gap()})
        info['has_solution'] = True
    except CplexError:
        pass

    return info


def solution_df(cpx, names = None):
    """
    create a data frame with the current solution for a CPLEX object (for debugging)
    :param cpx:
    :param names:
    :return:
    """
    assert isinstance(cpx, Cplex)
    if names is None:
        names = cpx.variables.get_names()
    else:
        assert isinstance(names, dict)
        names = reduce(concat, names.values())

    if has_solution(cpx):
        all_values = cpx.solution.get_values(names)
    else:
        all_values = np.repeat(np.nan, len(names)).tolist()

    df = pd.DataFrame({
        'name': names,
        'value': all_values,
        'lb': cpx.variables.get_lower_bounds(names),
        'ub': cpx.variables.get_upper_bounds(names),
        })

    return df


# General
def copy_cplex(cpx):
    cpx_copy = Cplex(cpx)
    cpx_parameters = cpx.parameters.get_changed()
    for (pname, pvalue) in cpx_parameters:
        phandle = reduce(getattr, str(pname).split("."), cpx_copy)
        phandle.set(pvalue)
    return cpx_copy


def get_lp_relaxation(cpx):
    rlx = copy_cplex(cpx)
    if rlx.get_problem_type() == rlx.problem_type.MILP:
        rlx.set_problem_type(rlx.problem_type.LP)
    return rlx


# Initialization
def add_mip_start(cpx, solution, effort_level = 1, name = None, indices = None):
    """
    :param cpx:
    :param solution:
    :param effort_level:    (must be one of the values of mip.MIP_starts.effort_level)
                            1 <-> check_feasibility
                            2 <-> solve_fixed
                            3 <-> solve_MIP
                            4 <-> repair
                            5 <-> no_check
    :param name:
    :param indices: CPLEX variable indices or names for each entry of solution;
                    None means solution[j] is the value of CPLEX variable j
    :return: mip
    """
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()

    if indices is None:
        indices = list(range(len(solution)))
    else:
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        indices = list(indices)
        assert len(indices) == len(solution), f"len(indices) = {len(indices)} != len(solution) = {len(solution)}"
        if all(isinstance(j, str) for j in indices):
            indices = cpx.variables.get_indices(indices)
        indices = np.array(indices, dtype = int).tolist()

    mip_start = SparsePair(val = solution, ind = indices)
    if name is None:
        cpx.MIP_starts.add(mip_start, effort_level)
    else:
        cpx.MIP_starts.add(mip_start, effort_level, name)

    return cpx


# Parameter Manipulation
def set_mip_parameters(cpx, param = CPX_MIP_PARAMETERS):

    # get parameter handle
    p = cpx.parameters

    # Record calls to C API
    # cpx.parameters.record.set(True)

    if param['display_cplex_progress'] in (None, False):
        cpx = set_cpx_display_options(cpx, display_mip = False, display_lp =  False, display_parameters = False)

    # major parameters
    p.randomseed.set(param['randomseed'])
    p.threads.set(param['n_cores'])
    p.output.clonelog.set(0)

    # solution strategy
    p.emphasis.mip.set(param['mipemphasis'])
    p.preprocessing.boundstrength.set(param['bound_strengthening'])
    p.read.scale.set(param['scale_parameters'])

    # cuts
    p.mip.cuts.implied.set(param['implied_bound_cuts'])
    p.mip.cuts.localimplied.set(param['locally_implied_bound_cuts'])
    p.mip.cuts.zerohalfcut.set(param['zero_half_cuts'])
    p.mip.cuts.mircut.set(param['mir_cuts'])
    p.mip.cuts.covers.set(param['cover_cuts'])
    #
    # tolerances
    p.emphasis.numerical.set(param['numerical_emphasis'])
    p.mip.tolerances.integrality.set(param['integrality_tolerance'])

    # initialization
    p.mip.limits.repairtries.set(param['repairtries'])

    # solution pool
    p.mip.pool.capacity.set(param['poolsize'])
    p.mip.pool.replace.set(param['poolreplace'])
    if not np.isnan(param['poolrelgap']):
        p.mip.pool.relgap.set(param['poolrelgap'])
    #
    # p.preprocessing.aggregator.set(0)
    # p.preprocessing.reduce.set(0)
    # p.preprocessing.presolve.set(0)
    # p.preprocessing.coeffreduce.set(0)
    # p.preprocessing.boundstrength.set(0)

    # stopping
    p.mip.tolerances.mipgap.set(param['mipgap'])
    p.mip.tolerances.absmipgap.set(param['absmipgap'])
    p.mip.tolerances.objdifference.set(param['objdifference'])

    if param['time_limit'] < CPX_MIP_PARAMETERS['time_limit']:
        cpx = set_mip_time_limit(cpx, param['time_limit'])

    if param['node_limit'] < CPX_MIP_PARAMETERS['node_limit']:
        cpx = set_mip_node_limit(cpx, param['node_limit'])

    # node file
    p.mip.limits.treememory.set(param['nodefilesize'])
    # p.workdir.Cur  = exp_workdir;
    # p.workmem.Cur                    = cplex_workingmem;
    # p.mip.strategy.file.Cur          = 2; %nodefile uncompressed

    return cpx


def get_mip_parameters(cpx):

    p = cpx.parameters

    param = {
        # major
        'display_cplex_progress': p.mip.display.get() > 0,
        'randomseed': p.randomseed.get(),
        'n_cores': p.threads.get(),
        #
        # strategy
        'mipemphasis': p.emphasis.mip.get(),
        'bound_strengthening': p.preprocessing.boundstrength.get(),
        'scale_parameters': p.read.scale.get(),
        #
        # cuts
        'implied_bound_cuts': p.mip.cuts.implied.get(),
        'locally_implied_bound_cuts': p.mip.cuts.localimplied.get(),
        'zero_half_cuts': p.mip.cuts.zerohalfcut.get(),
        'mir_cuts': p.mip.cuts.mircut.get(),
        'cover_cuts': p.mip.cuts.covers.get(),
        #
        # stopping
        'time_limit': p.timelimit.get(),
        'node_limit': p.mip.limits.nodes.get(),
        'mipgap': p.mip.tolerances.mipgap.get(),
        'absmipgap': p.mip.tolerances.absmipgap.get(),
        'objdifference': p.mip.tolerances.objdifference.get(),
        #
        # mip tolerances
        'integrality_tolerance': p.mip.tolerances.integrality.get(),
        'numerical_emphasis': p.emphasis.numerical.get(),
        #
        # solution pool
        'repairtries': p.mip.limits.repairtries.get(),
        'poolsize': p.mip.pool.capacity.get(),
        'poolreplace': p.mip.pool.replace.get(),
        'poolrelgap': p.mip.pool.relgap.get(),
        #
        # node file
        'nodefilesize': p.mip.limits.treememory.get(),
        # mip.parameters.workdir.Cur  = exp_workdir;
        # mip.parameters.workmem.Cur                    = cplex_workingmem;
        # mip.parameters.mip.strategy.file.Cur          = 2; %nodefile uncompressed
        }

    return param


def toggle_mip_preprocessing(cpx, toggle = True):
    """toggles pre-processing on/off for debugging / computational experiments"""

    # presolve
    # mip.parameters.preprocessing.presolve.help()
    # 0 = off
    # 1 = on

    # boundstrength
    # type of bound strengthening  :
    # -1 = automatic
    # 0 = off
    # 1 = on

    # reduce
    # mip.parameters.preprocessing.reduce.help()
    # type of primal and dual reductions  :
    # 0 = no primal and dual reductions
    # 1 = only primal reductions
    # 2 = only dual reductions
    # 3 = both primal and dual reductions

    # coeffreduce strength
    # level of coefficient reduction  :
    #   -1 = automatic
    #   0 = none
    #   1 = reduce only to integral coefficients
    #   2 = reduce any potential coefficient
    #   3 = aggressive reduction with tilting

    # dependency
    # indicator for preprocessing dependency checker  :
    #   -1 = automatic
    #   0 = off
    #   1 = at beginning
    #   2 = at end
    #   3 = at both beginning and end

    if toggle:
        cpx.parameters.preprocessing.aggregator.reset()
        cpx.parameters.preprocessing.reduce.reset()
        cpx.parameters.preprocessing.presolve.reset()
        cpx.parameters.preprocessing.coeffreduce.reset()
        cpx.parameters.preprocessing.boundstrength.reset()
    else:
        cpx.parameters.preprocessing.aggregator.set(0)
        cpx.parameters.preprocessing.reduce.set(0)
        cpx.parameters.preprocessing.presolve.set(0)
        cpx.parameters.preprocessing.coeffreduce.set(0)
        cpx.parameters.preprocessing.boundstrength.set(0)

    return cpx


def set_mip_cutoff_values(cpx, objval, objval_increment):
    """

    :param cpx:
    :param objval:
    :param objval_increment:
    :return:
    """
    assert objval_increment >= 0.0
    p = cpx.parameters
    if cpx.objective.get_sense() == cpx.objective.sense.maximize:
        p.mip.tolerances.lowercutoff.set(float(objval))
    else:
        assert objval >= 0.0
        p.mip.tolerances.uppercutoff.set(float(objval))
    p.mip.tolerances.objdifference.set(0.95 * float(objval_increment))
    p.mip.tolerances.absmipgap.set(0.95 * float(objval_increment))
    return cpx


# Display Options
def set_cpx_display_options(cpx, display_mip = True, display_parameters = False, display_lp = False):

    cpx.parameters.mip.display.set(display_mip)
    cpx.parameters.simplex.display.set(display_lp)
    cpx.parameters.paramdisplay.set(display_parameters)

    if not (display_mip or display_lp):
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        cpx.set_error_stream(None)
        cpx.set_warning_stream(None)

    return cpx


# Stopping Conditions
def set_mip_max_gap(cpx, max_gap = None):
    """
    sets the largest value of the relative optimality gap required to stop solving a MIP
    :param cpx:
    :param max_gap:
    :return:
    """
    if max_gap is not None:
        max_gap = float(max_gap)
        max_gap = min(max_gap, cpx.parameters.mip.tolerances.mipgap.max())
    else:
        max_gap = cpx.parameters.mip.tolerances.mipgap.min()

    assert max_gap >= 0.0
    cpx.parameters.mip.tolerances.mipgap.set(max_gap)

    return cpx


def set_mip_time_limit(cpx, time_limit = None):
    """

    :param cpx:
    :param time_limit:
    :return:
    """
    max_time_limit = float(cpx.parameters.timelimit.max())

    if time_limit is None:
        time_limit = max_time_limit
    else:
        time_limit = float(time_limit)
        time_limit = min(time_limit, max_time_limit)

    assert time_limit >= 0.0
    cpx.parameters.timelimit.set(time_limit)
    return cpx


def set_mip_node_limit(cpx, node_limit = None):
    """

    :param cpx:
    :param node_limit:
    :return:
    """
    max_node_limit = cpx.parameters.mip.limits.nodes.max()
    if node_limit is not None:
        node_limit = int(node_limit)
        node_limit = min(node_limit, max_node_limit)
    else:
        node_limit = max_node_limit

    assert node_limit >= 0.0
    cpx.parameters.mip.limits.nodes.set(node_limit)
    return cpx


#### Solution Pool ######
class SolutionPool:
    """
    helper class used to create/manipulate a queue of solutions and objective values
    """

    names = ['solution', 'coefficients', 'objval']

    def __init__(self, df = None, sense = "minimize"):

        assert sense in ("minimize", "maximize"), f"sense must be 'minimize' or 'maximize', got {sense!r}"
        self.sense = sense
        if df is None:
            self._df = pd.DataFrame(columns = self.names)
        else:
            assert isinstance(df, pd.DataFrame)
            self._df = df.copy()[self.names]

    def includes(self, solution):
        """
        :param solution: solution vector
        :return: True if there exists another solution in this object that matches th
        """
        return any(solution == old for old in self._df['solution'])

    def add(self, solution, coefficients, objval):
        """
        :param solution:
        :param coefficients:
        :param objval:
        :return:
        """

        if isinstance(objval, (list, np.ndarray)):
            # convert placeholder for prediction constraints to list of appropriate size
            assert all(len(param) == len(objval) for param in (solution, coefficients))
            param_dict = {'solution': solution,
                          'objval': objval,
                          'coefficients': coefficients}
        else:
            param_dict = {'solution': [solution],
                          'objval': [objval],
                          'coefficients': [coefficients]}

        new_df = pd.DataFrame.from_dict(param_dict)
        self._df = pd.concat([self._df, new_df]).reset_index(drop = True)


    def get_best_solution(self):
        if self.sense == "maximize":
            best_idx = self._df['objval'].idxmax()
        else:
            best_idx = self._df['objval'].idxmin()
        best_solution = self._df.iloc[best_idx].to_dict()
        return best_solution

    def merge(self, pool):
        other = pool._df if isinstance(pool, SolutionPool) else pool
        self._df = pd.concat([self._df, other], sort=False).reset_index(drop=True)

    def get_df(self):
        return self._df.copy(deep=True)

    def clear(self):
        self._df.drop(self._df.index, inplace=True)

    @property
    def size(self):
        return self._df.shape[0]

    @property
    def objvals(self):
        return self._df['objval'].tolist()

    @property
    def solutions(self):
        return self._df['solution'].tolist()

    @property
    def coefficients(self):
        return self._df['coefficients'].tolist()

    def __len__(self):
        return len(self._df)

    def __repr__(self):
        return self._df.__repr__()

    def __str__(self):
        return self._df.__str__()


# ---- debugging helpers
def is_integer(x):
    return np.array_equal(x, np.require(x, dtype=np.int_))


def is_binary(x):
    return np.array_equal(x, np.require(x, dtype=np.bool_))


def check_variable_type(val, type):
    if type == 'B':
        return is_binary(val)
    elif type == 'I':
        return is_integer(val)
    elif type == 'C':
        return np.isfinite(val)
    else:
        return False


def compare_with_incumbent_solution(cpx, solution, indices=None):
    """
    compares solution vector to incumbent solution of cpx object
    :param cpx:
    :param solution:
    :param indices:
    :return:
    """

    if indices is None:
        expected_solution = np.array(cpx.solution.get_values())
    else:
        expected_solution = np.array(cpx.solution.get_values(indices))

    diff_idx = np.flatnonzero(solution != expected_solution)

    if len(diff_idx) > 0:
        diff_names = cpx.variables.get_names(diff_idx.tolist())
    else:
        diff_names = []

    return diff_idx, diff_names


def print_score_error(bug_idx, score_values, theta, U, Y, L, class_label="pos"):
    out = ""
    for i in bug_idx:
        u_vals = ", ".join(f"{U[i, j]:1.5f}" for j in range(len(theta)))
        theta_vals = ", ".join(f"{theta[j]:1.5f}" for j in range(len(theta)))
        out += f"\nU_{class_label}[{i},:] \t=\t({u_vals})\n"
        out += f"theta \t\t=\t({theta_vals})\n"
        out += f"score[{i}]  \t=\t{score_values[i]:1.5f}\n\n"
        out += f"y_hat[{i}]  \t=\t{int(np.sign(score_values[i]))}\n"
        out += f"y_true[{i}] \t=\t{Y[i]}\n"
        out += f"l_true[{i}] \t=\t{int(Y[i] != np.sign(score_values[i]))}\n"
        out += f"l_cplex[{i}] \t=\t{L[i]}\n\n"
    return out
