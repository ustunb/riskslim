"""CPLEX callbacks."""

import time
import numpy as np

from cplex.callbacks import HeuristicCallback, LazyConstraintCallback

from riskslim.utils import Stats, cast_to_integer, is_integer
from riskslim.mip import convert_to_risk_slim_cplex_solution
from riskslim.solution_pool import FastSolutionPool
from riskslim.bounds import chained_updates
from riskslim.utils import print_log


class LossCallback(LazyConstraintCallback):
    """
    Call initialize() before using this callback.

    LossCallback is called when CPLEX finds an integer feasible solution. By default, it will add a cut at this
    solution to improve the cutting-plane approximation of the loss function. The cut is added as a 'lazy' constraint
    into the surrogate LP so that it is evaluated only when necessary.

    Optional Functionality:

    - add an initial set of cutting planes found by warm starting
      requires initial_cuts

    - pass integer feasible solutions to 'polish' queue so that they can be polished with DCD in the PolishAndRoundCallback
      requires settings['polish_flag'] = True

    - adds cuts at integer feasible solutions found by the PolishAndRoundCallback
      requires settings['add_cuts_at_heuristic_solutions'] = True

    - reduces overall search region by adding constraints on objval_max, l0_max, loss_min, loss_max
      requires settings['chained_updates_flag'] = True
    """

    def initialize(
            self, indices, stats, settings, compute_loss_cut, get_alpha,
            get_L0_penalty_from_alpha, initial_cuts=None, cut_queue=None, polish_queue=None,
            verbose=True
        ):

        assert isinstance(indices, dict)
        assert isinstance(stats, Stats)
        assert isinstance(settings, dict)
        assert callable(compute_loss_cut)
        assert callable(get_alpha)
        assert callable(get_L0_penalty_from_alpha)


        self.settings = settings  #store pointer to shared settings so that settings can be turned on/off during B&B
        self.stats = stats  # dict containing information for flow
        self.verbose = verbose

        # todo (validate initial cutting planes)
        self.initial_cuts = initial_cuts

        # indices
        self.rho_idx = indices['rho']
        self.cut_idx = indices['loss'] + indices['rho']
        self.alpha_idx = indices['alpha']
        self.L0_reg_ind = indices['L0_reg_ind']
        self.C_0_nnz = indices['C_0_nnz']
        self.compute_loss_cut = compute_loss_cut
        self.get_alpha = get_alpha
        self.get_L0_penalty_from_alpha = get_L0_penalty_from_alpha

        # cplex can drop cutting planes that are not used. by default, we force CPLEX to use all cutting planes.
        self.loss_cut_purge_flag = self.use_constraint.purge if self.settings['purge_loss_cuts'] else self.use_constraint.force

        # setup pointer to cut_queue to receive cuts from PolishAndRoundCallback
        if self.settings['add_cuts_at_heuristic_solutions']:
            if cut_queue is None:
                self.cut_queue = FastSolutionPool(len(self.rho_idx))
            else:
                assert isinstance(cut_queue, FastSolutionPool)
                self.cut_queue = cut_queue

        # setup pointer to polish_queue to send integer solutions to PolishAndRoundCallback
        if self.settings['polish_flag']:
            if polish_queue is None:
                self.polish_queue = FastSolutionPool(len(self.rho_idx))
            else:
                assert isinstance(polish_queue, FastSolutionPool)
                self.polish_queue = polish_queue

        # setup indices for chained_updates
        if self.settings['chained_updates_flag']:
            self.loss_cut_constraint = [indices['loss'], [1.0]]
            self.L0_cut_constraint = [[indices['L0_norm']], [1.0]]
            self.objval_cut_constraint = [[indices['objval']], [1.0]]
            self.bound_cut_purge_flag = self.use_constraint.purge if self.settings['purge_loss_cuts'] else self.use_constraint.force

        return

    def add_loss_cut(self, rho):

        loss_value, loss_slope = self.compute_loss_cut(rho)
        self.add(constraint = [self.cut_idx, [1.0] + (-loss_slope).tolist()],
                 sense = "G",
                 rhs = float(loss_value - loss_slope.dot(rho)),
                 use = self.loss_cut_purge_flag)

        return loss_value

    def update_bounds(self):
        """

        Returns
        -------

        """

        bounds = chained_updates(bounds = self.stats.bounds,
                                 C_0_nnz = self.C_0_nnz,
                                 new_objval_at_relaxation = self.stats.lowerbound,
                                 new_objval_at_feasible = self.stats.upperbound)

        #add cuts if bounds need to be tighter
        if bounds.loss_min > self.stats.bounds.loss_min:
            self.add(constraint = self.loss_cut_constraint, sense = "G", rhs = bounds.loss_min, use = self.bound_cut_purge_flag)
            self.stats.bounds.loss_min = bounds.loss_min
            self.stats.n_bound_updates_loss_min += 1

        if bounds.objval_min > self.stats.bounds.objval_min:
            self.add(constraint = self.objval_cut_constraint, sense = "G", rhs = bounds.objval_min, use = self.bound_cut_purge_flag)
            self.stats.bounds.objval_min = bounds.objval_min
            self.stats.n_bound_updates_objval_min += 1

        if bounds.max_size < self.stats.bounds.max_size:
            self.add(constraint = self.L0_cut_constraint, sense="L", rhs = bounds.max_size, use = self.bound_cut_purge_flag)
            self.stats.bounds.max_size = bounds.max_size
            self.stats.n_bound_updates_max_size += 1

        if bounds.loss_max < self.stats.bounds.loss_max:
            self.add(constraint = self.loss_cut_constraint, sense="L", rhs = bounds.loss_max, use = self.bound_cut_purge_flag)
            self.stats.bounds.loss_max = bounds.loss_max
            self.stats.n_bound_updates_loss_max += 1

        if bounds.objval_max < self.stats.bounds.objval_max:
            self.add(constraint = self.objval_cut_constraint, sense="L", rhs = bounds.objval_max, use = self.bound_cut_purge_flag)
            self.stats.bounds.objval_max = bounds.objval_max
            self.stats.n_bound_updates_objval_max += 1

        return

    def __call__(self):

        #log('in cut callback')
        callback_start_time = time.time()

        #record entry metrics
        self.stats.cut_callback_times_called += 1
        self.stats.lowerbound = self.get_best_objective_value()
        self.stats.relative_gap = self.get_MIP_relative_gap()
        self.stats.nodes_processed = self.get_num_nodes()
        self.stats.nodes_remaining = self.get_num_remaining_nodes()

        # add initial cuts first time the callback is used
        if self.initial_cuts is not None:
            print_log('adding %1.0f initial cuts' % len(self.initial_cuts['lhs']), self.verbose)
            for cut, lhs in zip(self.initial_cuts['coefs'], self.initial_cuts['lhs']):
                self.add(constraint = cut, sense = "G", rhs = lhs, use = self.loss_cut_purge_flag)
            self.initial_cuts = None

        # get integer feasible solution
        rho = np.array(self.get_values(self.rho_idx))
        alpha = np.array(self.get_values(self.alpha_idx))

        # check that CPLEX is actually integer. if not, then recast as int
        if not is_integer(rho):
            rho = cast_to_integer(rho)
            alpha = self.get_alpha(rho)

        # add cutting plane at integer feasible solution
        cut_start_time = time.time()
        loss_value = self.add_loss_cut(rho)
        cut_time = time.time() - cut_start_time
        cuts_added = 1

        # if solution updates incumbent, then add solution to queue for polishing
        current_upperbound = float(loss_value + self.get_L0_penalty_from_alpha(alpha))
        incumbent_update = current_upperbound < self.stats.upperbound

        if incumbent_update:
            self.stats.incumbent = rho
            self.stats.upperbound = current_upperbound
            self.stats.n_incumbent_updates += 1

        if self.settings['polish_flag']:
            polishing_cutoff = self.stats.upperbound * (1.0 + self.settings['polishing_tolerance'])
            if current_upperbound < polishing_cutoff:
                self.polish_queue.add(current_upperbound, rho)

        # add cutting planes at other integer feasible solutions in cut_queue
        if self.settings['add_cuts_at_heuristic_solutions']:
            if len(self.cut_queue) > 0:
                self.cut_queue.filter_sort_unique()
                cut_start_time = time.time()
                for cut_rho in self.cut_queue.solutions:
                    self.add_loss_cut(cut_rho)
                cut_time += time.time() - cut_start_time
                cuts_added += len(self.cut_queue)
                self.cut_queue.clear()

        # update bounds
        if self.settings['chained_updates_flag']:
            if (self.stats.lowerbound > self.stats.bounds.objval_min) or (self.stats.upperbound < self.stats.bounds.objval_max):
                self.stats.n_update_bounds_calls += 1
                self.update_bounds()

        # record metrics at end
        self.stats.n_cuts += cuts_added
        self.stats.total_cut_time += cut_time
        self.stats.total_cut_callback_time += time.time() - callback_start_time

        #log('left cut callback')
        return


class PolishAndRoundCallback(HeuristicCallback):
    """
    Call initialize() before using this callback.

    HeuristicCallback is called intermittently during B&B to generate new feasible
    solutions by rounding, and to improve the quality of existing feasible solutions
    by polishing. Specific heuristics include:

    - Sequential Rounding on the continuous solution from the surrogate LP (only if there has been a change in the
      lower bound). Requires settings['round_flag'] = True. If settings['polish_after_rounding'] = True, then the
      rounded solutions are polished using DCD.

    - DCD Polishing: Polishes integer solutions in polish_queue using DCD. Requires settings['polish_flag'] = True.

    Optional Functionality:

    - Feasible solutions are passed to LazyCutConstraintCallback via cut_queue

    Known issues:

    - Sometimes CPLEX does not return an integer feasible solution (in which case we correct this manually)
    """

    def initialize(self, indices, control, settings, cut_queue, polish_queue, get_objval, get_L0_norm, is_feasible, polishing_handle, rounding_handle):

        #todo: add basic assertions to make sure that nothing weird is going on
        assert isinstance(indices, dict)
        assert isinstance(control, Stats)
        assert isinstance(settings, dict)
        assert isinstance(cut_queue, FastSolutionPool)
        assert isinstance(polish_queue, FastSolutionPool)
        assert callable(get_objval)
        assert callable(get_L0_norm)
        assert callable(is_feasible)
        assert callable(polishing_handle)
        assert callable(rounding_handle)

        self.rho_idx = indices['rho']
        self.L0_reg_ind = indices['L0_reg_ind']
        self.C_0_nnz = indices['C_0_nnz']
        self.indices = indices
        self.previous_lowerbound = 0.0
        self.stats = control
        self.settings = settings

        self.round_flag = settings['round_flag']
        self.polish_rounded_solutions = settings['polish_rounded_solutions']
        self.polish_flag = settings['polish_flag']
        self.polish_queue = polish_queue

        self.cut_queue = cut_queue  # pointer to cut_queue

        self.rounding_tolerance = float(1.0 + settings['rounding_tolerance'])
        self.rounding_start_cuts = settings['rounding_start_cuts']
        self.rounding_stop_cuts = settings['rounding_stop_cuts']
        self.rounding_stop_gap = settings['rounding_stop_gap']
        self.rounding_start_gap = settings['rounding_start_gap']

        self.polishing_tolerance = float(1.0 + settings['polishing_tolerance'])
        self.polishing_start_cuts = settings['polishing_start_cuts']
        self.polishing_stop_cuts = settings['polishing_stop_cuts']
        self.polishing_stop_gap = settings['polishing_stop_gap']
        self.polishing_start_gap = settings['polishing_start_gap']
        self.polishing_max_solutions = settings['polishing_max_solutions']
        self.polishing_max_runtime = settings['polishing_max_runtime']

        self.get_objval = get_objval
        self.get_L0_norm = get_L0_norm
        self.is_feasible = is_feasible
        self.polishing_handle = polishing_handle
        self.rounding_handle = rounding_handle

        return


    def update_heuristic_flags(self, n_cuts, relative_gap):

        # keep on rounding?
        keep_rounding = (self.rounding_start_cuts <= n_cuts <= self.rounding_stop_cuts) and \
                        (self.rounding_stop_gap <= relative_gap <= self.rounding_start_gap)

        # keep on polishing?
        keep_polishing = (self.polishing_start_cuts <= n_cuts <= self.polishing_stop_cuts) and \
                         (self.polishing_stop_gap <=  relative_gap <= self.polishing_start_gap)

        self.round_flag &= keep_rounding
        self.polish_flag &= keep_polishing
        self.polish_rounded_solutions &= self.round_flag

        return


    def __call__(self):
        # todo write rounding/polishing as separate function calls

        # log('in heuristic callback')
        if not (self.round_flag or self.polish_flag):
            return

        callback_start_time = time.time()
        self.stats.heuristic_callback_times_called += 1
        self.stats.upperbound = self.get_incumbent_objective_value()
        self.stats.lowerbound = self.get_best_objective_value()
        self.stats.relative_gap = self.get_MIP_relative_gap()

        # check if lower bound was updated since last call
        lowerbound_update = self.previous_lowerbound < self.stats.lowerbound
        if lowerbound_update:
            self.previous_lowerbound = self.stats.lowerbound

        # check if incumbent solution has been updated
        # if incumbent solution is not integer, then recast as integer and update objective value manually
        if self.has_incumbent():
            cplex_incumbent = np.array(self.get_incumbent_values(self.rho_idx))
            cplex_rounding_issue = not is_integer(cplex_incumbent)
            if cplex_rounding_issue:
                cplex_incumbent = cast_to_integer(cplex_incumbent)

            incumbent_update = not np.array_equal(cplex_incumbent, self.stats.incumbent)

            if incumbent_update:
                self.stats.incumbent = cplex_incumbent
                self.stats.n_incumbent_updates += 1
                if cplex_rounding_issue:
                    self.stats.upperbound = self.get_objval(cplex_incumbent)

        # update flags on whether or not to keep rounding / polishing
        self.update_heuristic_flags(n_cuts = self.stats.n_cuts, relative_gap = self.stats.relative_gap)

        #variables to store best objective value / solution from heuristics
        best_objval = float('inf')
        best_solution = None

        # run sequential rounding if lower bound was updated since the last call
        if self.round_flag and lowerbound_update:

            rho_cts = np.array(self.get_values(self.rho_idx))
            zero_idx_rho_ceil = np.equal(np.ceil(rho_cts), 0)
            zero_idx_rho_floor = np.equal(np.floor(rho_cts), 0)
            cannot_round_to_zero = np.logical_not(np.logical_or(zero_idx_rho_ceil, zero_idx_rho_floor))
            min_l0_norm = np.count_nonzero(cannot_round_to_zero[self.L0_reg_ind])
            max_l0_norm = np.count_nonzero(rho_cts[self.L0_reg_ind])
            rounded_solution_is_feasible = (min_l0_norm < self.stats.bounds.max_size and max_l0_norm > self.stats.bounds.min_size)

            if rounded_solution_is_feasible:

                rounding_cutoff = self.rounding_tolerance * self.stats.upperbound
                rounding_start_time = time.time()
                rounded_solution, rounded_objval, early_stop = self.rounding_handle(rho_cts, rounding_cutoff)
                self.stats.total_round_time += time.time() - rounding_start_time
                self.stats.n_rounded += 1

                # round solution if sequential rounding did not stop early
                if not early_stop:

                    if self.settings['add_cuts_at_heuristic_solutions']:
                        self.cut_queue.add(rounded_objval, rounded_solution)

                    if self.is_feasible(rounded_solution):
                        best_solution = rounded_solution
                        best_objval = rounded_objval

                    if self.polish_rounded_solutions:
                        current_upperbound = min(rounded_objval, self.stats.upperbound)
                        polishing_cutoff = current_upperbound * self.polishing_tolerance

                        if rounded_objval < polishing_cutoff:
                            start_time = time.time()
                            polished_solution, _, polished_objval = self.polishing_handle(rounded_solution)
                            self.stats.total_round_then_polish_time += time.time() - start_time
                            self.stats.n_rounded_then_polished += 1

                            if self.settings['add_cuts_at_heuristic_solutions']:
                                self.cut_queue.add(polished_objval, polished_solution)

                            if self.is_feasible(polished_solution):
                                best_solution = polished_solution
                                best_objval = polished_objval

        # polish solutions in polish_queue or that were produced by rounding
        if self.polish_flag and len(self.polish_queue) > 0:

            #get current upperbound
            current_upperbound = min(best_objval, self.stats.upperbound)
            polishing_cutoff = self.polishing_tolerance * current_upperbound
            self.polish_queue.filter_sort_unique(max_objval = polishing_cutoff)

            if len(self.polish_queue) > 0:
                polished_queue = FastSolutionPool(self.polish_queue.P)
                polish_time = 0
                n_polished = 0
                for objval, solution in zip(self.polish_queue.objvals, self.polish_queue.solutions):

                    if objval >= polishing_cutoff:
                        break

                    polish_start_time = time.time()
                    polished_solution, _, polished_objval = self.polishing_handle(solution)
                    polish_time += time.time() - polish_start_time
                    n_polished += 1

                    # update cutoff whenever the solution returns an infeasible solutions
                    if self.is_feasible(polished_solution):
                        polished_queue.add(polished_objval, polished_solution)
                        current_upperbound = min(polished_objval, polished_objval)
                        polishing_cutoff = self.polishing_tolerance * current_upperbound

                    if polish_time > self.polishing_max_runtime:
                        break

                    if n_polished > self.polishing_max_solutions:
                        break

                self.polish_queue.clear()
                self.stats.total_polish_time += polish_time
                self.stats.n_polished += n_polished

                if self.settings['add_cuts_at_heuristic_solutions']:
                    self.cut_queue.add(polished_queue.objvals, polished_queue.solutions)

                # check if the best polished solution will improve the queue
                polished_queue.filter_sort_unique(max_objval = best_objval)
                if len(polished_queue) > 0:
                    best_objval, best_solution = polished_queue.get_best_objval_and_solution()

        # if heuristics produces a better solution then update the incumbent
        heuristic_update = best_objval < self.stats.upperbound
        if heuristic_update:
            self.stats.n_heuristic_updates += 1
            proposed_solution, proposed_objval = convert_to_risk_slim_cplex_solution(indices = self.indices, rho = best_solution, objval = best_objval)
            self.set_solution(solution = proposed_solution, objective_value = proposed_objval)

        self.stats.total_heuristic_callback_time += time.time() - callback_start_time
        #log('left heuristic callback')
        return
