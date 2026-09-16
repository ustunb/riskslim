import numpy as np
from functools import reduce
from cplex import Cplex, SparsePair
from cplex.exceptions import CplexError
from cplex.callbacks import MIPInfoCallback


CPX_MIP_PARAMETERS = {
    #
    'display_cplex_progress': True,
    #set to True to show CPLEX progress in console
    #
    'n_cores': 1,
    # Number of CPU cores to use in B & B
    # May have to set n_cores = 1 in order to use certain stats callbacks in CPLEX 12.7.0 and earlier
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
    # -1    = cplex chooses
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
        info.update({'values': np.array(sol.get_values()),
                     'objval': sol.get_objective_value(),
                     'upperbound': sol.MIP.get_cutoff(),
                     'lowerbound': sol.MIP.get_best_objective(),
                     'gap': sol.MIP.get_mip_relative_gap()})
        info['has_solution'] = True
    except CplexError:
        pass

    return info


# Initialization
def add_mip_start(cpx, solution, effort_level = 1, name = None):
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
    :return: mip
    """
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()

    mip_start = SparsePair(val = solution, ind = list(range(len(solution))))
    if name is None:
        cpx.MIP_starts.add(mip_start, effort_level)
    else:
        cpx.MIP_starts.add(mip_start, effort_level, name)

    return cpx


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
    if rlx.get_problem_type() is rlx.problem_type.MILP:
        rlx.set_problem_type(rlx.problem_type.LP)
    return rlx


# Stamping
def add_variable(cpx, name, obj, ub, lb, vtype):

    assert isinstance(cpx, Cplex)

    # name
    if isinstance(name, np.ndarray):
        name = name.tolist()
    elif isinstance(name, str):
        name = [name]

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
                raise ValueError('invalid length: len(vtype) = %d. expected either 1 or %d' % (len(vtype), nvars))

        if isinstance(obj, np.ndarray):
            obj = obj.astype(float).tolist()
        elif isinstance(obj, list):
            if len(obj) == nvars:
                obj = [float(v) for v in obj]
            elif len(obj) == 1:
                obj = nvars * [float(obj)]
            else:
                raise ValueError('invalid length: len(obj) = %d. expected either 1 or %d' % (len(obj), nvars))
        else:
            obj = nvars * [float(obj)]

        if isinstance(ub, np.ndarray):
            ub = ub.astype(float).tolist()
        elif isinstance(ub, list):
            if len(ub) == nvars:
                ub = [float(v) for v in ub]
            elif len(ub) == 1:
                ub = nvars * [float(ub)]
            else:
                raise ValueError('invalid length: len(ub) = %d. expected either 1 or %d' % (len(ub), nvars))
        else:
            ub = nvars * [float(ub)]

        if isinstance(lb, np.ndarray):
            lb = lb.astype(float).tolist()
        elif isinstance(lb, list):
            if len(lb) == nvars:
                lb = [float(v) for v in lb]
            elif len(ub) == 1:
                lb = nvars * [float(lb)]
            else:
                raise ValueError('invalid length: len(lb) = %d. expected either 1 or %d' % (len(lb), nvars))
        else:
            lb = nvars * [float(lb)]

    # check that all components are lists
    assert isinstance(name, list)
    assert isinstance(obj, list)
    assert isinstance(ub, list)
    assert isinstance(lb, list)
    assert isinstance(vtype, list)

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


# Parameter Manipulation
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


def set_mip_parameters(cpx, param = CPX_MIP_PARAMETERS):

    # get parameter handle
    p = cpx.parameters

    # Record calls to C API
    # cpx.parameters.record.set(True)

    if param['display_cplex_progress'] is (None or False):
        cpx = set_cpx_display_options(cpx, display_mip = False, display_lp = False, display_parameters = False)

    # major parameters
    p.randomseed.set(param['randomseed'])
    p.output.clonelog.set(0)

    # solution strategy
    p.emphasis.mip.set(param['mipemphasis'])
    p.preprocessing.boundstrength.set(param['bound_strengthening'])

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
    #
    # p.preprocessing.aggregator.set(0)
    # p.preprocessing.reduce.set(0)
    # p.preprocessing.presolve.set(0)
    # p.preprocessing.coeffreduce.set(0)
    # p.preprocessing.boundstrength.set(0)

    # stopping
    p.mip.tolerances.mipgap.set(param['mipgap'])
    p.mip.tolerances.absmipgap.set(param['absmipgap'])

    if param['time_limit'] < CPX_MIP_PARAMETERS['time_limit']:
        cpx = set_mip_time_limit(cpx, param['time_limit'])

    if param['node_limit'] < CPX_MIP_PARAMETERS['node_limit']:
        cpx = set_mip_node_limit(cpx, param['node_limit'])

    # node file
    # p.workdir.Cur  = exp_workdir;
    # p.workmem.Cur                    = cplex_workingmem;
    # p.mip.strategy.file.Cur          = 2; %nodefile uncompressed
    # p.mip.limits.treememory.Cur      = cplex_nodefilesize;

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
        'scale_parameters': p.read.scale.get(),
        'locally_implied_bound_cuts': p.mip.cuts.localimplied.get(),
        #
        # stopping
        'time_limit': p.timelimit.get(),
        'node_limit': p.mip.limits.nodes.get(),
        'mipgap': p.mip.tolerances.mipgap.get(),
        'absmipgap': p.mip.tolerances.absmipgap.get(),
        #
        # mip tolerances
        'integrality_tolerance': p.mip.tolerances.integrality.get(),
        'numerical_emphasis': p.emphasis.numerical.get(),
        #
        # solution pool
        'repairtries': p.mip.limits.repairtries.get(),
        'poolsize': p.mip.pool.capacity.get(),
        'poolreplace': p.mip.pool.replace.get(),
        #
        # node file
        # mip.parameters.workdir.Cur  = exp_workdir;
        # mip.parameters.workmem.Cur                    = cplex_workingmem;
        # mip.parameters.mip.strategy.file.Cur          = 2; %nodefile uncompressed
        # mip.parameters.mip.limits.treememory.Cur      = cplex_nodefilesize;
        }

    return param


def set_mip_cutoff_values(cpx, objval, objval_increment):
    """

    :param cpx:
    :param objval:
    :param objval_increment:
    :return:
    """
    assert objval >= 0.0
    assert objval_increment >= 0.0
    p = cpx.parameters
    p.mip.tolerances.uppercutoff.set(float(objval))
    p.mip.tolerances.objdifference.set(0.95 * float(objval_increment))
    p.mip.tolerances.absmipgap.set(0.95 * float(objval_increment))
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


# Branch and Bound Statistics
class StatsCallback(MIPInfoCallback):


    def initialize(self, store_solutions = False, solution_start_idx = None, solution_end_idx = None):

        # scalars
        self.times_called = 0
        self.start_time = None

        # stats that are stored at every call len(stat) = times_called
        self.runtimes = []
        self.simplex_iterations = []
        self.nodes_processed = []
        self.nodes_remaining = []
        self.lowerbounds = []

        # stats that are stored at every incumbent update
        self.best_objval = float('inf')
        self.update_iterations = []
        self.incumbents = []
        self.upperbounds = []

        self.store_incumbent_solutions = store_solutions
        if self.store_incumbent_solutions:
            assert solution_start_idx <= solution_end_idx
            self.start_idx, self.end_idx = int(solution_start_idx), int(solution_end_idx)
            self.process_incumbent = self.record_objval_and_solution_before_incumbent
        else:
            self.process_incumbent = self.record_objval_before_incumbent


    def __call__(self):
        self.times_called += 1
        if self.start_time is None:
            self.start_time = self.get_start_time()
        self.runtimes.append(self.get_time())
        self.lowerbounds.append(self.get_best_objective_value())
        self.nodes_processed.append(self.get_num_nodes())
        self.nodes_remaining.append(self.get_num_remaining_nodes())
        self.simplex_iterations.append(self.get_num_iterations())
        self.process_incumbent()


    def record_objval_before_incumbent(self):
        if self.has_incumbent():
            self.record_objval()
            self.process_incumbent = self.record_objval


    def record_objval_and_solution_before_incumbent(self):
        if self.has_incumbent():
            self.record_objval_and_solution()
            self.process_incumbent = self.record_objval_and_solution


    def record_objval(self):
        objval = self.get_incumbent_objective_value()
        if objval < self.best_objval:
            self.best_objval = objval
            self.update_iterations.append(self.times_called)
            self.upperbounds.append(objval)


    def record_objval_and_solution(self):
        objval = self.get_incumbent_objective_value()
        if objval < self.best_objval:
            self.update_iterations.append(self.times_called)
            self.incumbents.append(self.get_incumbent_values(self.start_idx, self.end_idx))
            self.upperbounds.append(objval)
            self.best_objval = objval


    def check_stats(self):
        """checks stats rep at any point during the solution process"""

        # try:
        n_calls = len(self.runtimes)
        n_updates = len(self.upperbounds)
        assert n_updates <= n_calls

        if n_calls > 0:

            assert len(self.nodes_processed) == n_calls
            assert len(self.nodes_remaining) == n_calls
            assert len(self.lowerbounds) == n_calls

            lowerbounds = np.array(self.lowerbounds)
            for ub in self.upperbounds:
                assert np.greater_equal(ub, lowerbounds).all()

            runtimes = np.array(self.runtimes) - self.start_time
            nodes_processed = np.array(self.nodes_processed)

            is_increasing = lambda x: np.greater_equal(np.diff(x), 0.0).all()
            assert is_increasing(runtimes)
            assert is_increasing(nodes_processed)

        if n_updates > 0:

            assert len(self.update_iterations) == n_updates
            if self.store_incumbent_solutions:
                assert len(self.incumbents) == n_updates

            update_iterations = np.array(self.update_iterations)
            upperbounds = np.array(self.upperbounds)
            gaps = (upperbounds - lowerbounds[update_iterations - 1]) / (np.finfo(float).eps + upperbounds)

            is_increasing = lambda x: (np.diff(x) >= 0).all()
            assert is_increasing(update_iterations)
            assert is_increasing(-gaps)

        return True


    def get_stats(self, return_solutions = False):

        assert self.check_stats()
        import pandas as pd
        MAX_UPPERBOUND = float('inf')
        MAX_GAP = 1.00

        stats = pd.DataFrame({
            'runtime': [t - self.start_time for t in self.runtimes],
            'nodes_processed': list(self.nodes_processed),
            'nodes_remaining': list(self.nodes_remaining),
            'simplex_iterations': list(self.simplex_iterations),
            'lowerbound': list(self.lowerbounds)
            })

        upperbounds = list(self.upperbounds)
        update_iterations = list(self.update_iterations)
        incumbents = []  # empty placeholder

        # add upper bounds as well as iterations where the incumbent changes
        if update_iterations[0] > 1:
            update_iterations.insert(0, 1)
            upperbounds.insert(0, MAX_UPPERBOUND)
        row_idx = [i - 1 for i in update_iterations]

        stats = stats.assign(iterations = pd.Series(data = update_iterations, index = row_idx),
                             upperbound = pd.Series(data = upperbounds, index = row_idx))
        stats['incumbent_update'] = np.where(~np.isnan(stats['iterations']), True, False)
        stats = stats.fillna(method = 'ffill')

        # add relative gap
        gap = (stats['upperbound'] - stats['lowerbound']) / (stats['upperbound'] + np.finfo(float).eps)
        stats['gap'] = np.fmin(MAX_GAP, gap)

        # add model ids
        if return_solutions:
            incumbents = list(self.incumbents)
            model_ids = range(len(incumbents))
            row_idx = [i - 1 for i in self.update_iterations]
            stats = stats.assign(model_ids = pd.Series(data = model_ids, index = row_idx))
            stats = stats[['runtime',
                           'gap',
                           'upperbound',
                           'lowerbound',
                           'nodes_processed',
                           'nodes_remaining',
                           'simplex_iterations'
                           'model_id',
                           'incumbent_update']]

        else:
            stats = stats[['runtime',
                           'gap',
                           'upperbound',
                           'lowerbound',
                           'nodes_processed',
                           'nodes_remaining',
                           'simplex_iterations']]

        return stats, incumbents

