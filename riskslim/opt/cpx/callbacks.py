import numpy as np
import pandas as pd
from cplex.callbacks import MIPInfoCallback


# --- begin BLM StatsCallback (vendored; do not edit)
class StatsCallback(MIPInfoCallback):

    def initialize(self, store_solutions = False, solution_start_idx = None, solution_end_idx = None, sense = "minimize"):

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
        assert sense in ("minimize", "maximize"), f"sense must be 'minimize' or 'maximize', got {sense!r}"
        self.sense = sense
        self.best_objval = float('inf') if sense == "minimize" else float('-inf')
        self.update_iterations = []
        self.incumbents = []
        self.upperbounds = []
        self.store_solutions = store_solutions
        if self.store_solutions:
            assert solution_start_idx is not None and solution_end_idx is not None, \
                "store_solutions=True requires solution_start_idx and solution_end_idx"
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

    def improves_best_objval(self, objval):
        if self.sense == "minimize":
            return objval < self.best_objval
        return objval > self.best_objval

    def record_objval_before_incumbent(self):
        if self.has_incumbent():
            self.record_objval()
            self.process_incumbent = self.record_objval

    def record_objval(self):
        objval = self.get_incumbent_objective_value()
        if self.improves_best_objval(objval):
            self.update_iterations.append(self.times_called)
            self.best_objval = objval
            self.upperbounds.append(objval)

    def record_objval_and_solution_before_incumbent(self):
        if self.has_incumbent():
            self.record_objval_and_solution()
            self.process_incumbent = self.record_objval_and_solution

    def record_objval_and_solution(self):
        objval = self.get_incumbent_objective_value()
        if self.improves_best_objval(objval):
            self.update_iterations.append(self.times_called)
            self.best_objval = objval
            self.upperbounds.append(objval)
            self.incumbents.append(self.get_incumbent_values(self.start_idx, self.end_idx))

    def check_stats(self):
        """checks stats rep at any point during the solution process"""

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
            if self.store_solutions:
                assert len(self.incumbents) == n_updates
            update_iterations = np.array(self.update_iterations)
            upperbounds = np.array(self.upperbounds)
            gaps = (upperbounds - lowerbounds[update_iterations - 1]) / (np.finfo(float).tiny + upperbounds)
            is_increasing = lambda x: (np.diff(x) >= 0).all()
            assert is_increasing(update_iterations)
            assert is_increasing(-gaps)

        return True

    def get_stats(self):

        assert self.check_stats()
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

        # add upper bounds as well as iterations where the incumbent changes.
        # if CPLEX never found an incumbent (timed out before any feasible
        # solution), seed a single placeholder so ffill propagates inf and
        # the resulting gap is 1.0 (= MAX_GAP) for the whole trace.
        if not update_iterations:
            update_iterations = [1]
            upperbounds = [MAX_UPPERBOUND]
        elif update_iterations[0] > 1:
            update_iterations.insert(0, 1)
            upperbounds.insert(0, MAX_UPPERBOUND)
        row_idx = [i - 1 for i in update_iterations]
        stats = stats.assign(
                iterations = pd.Series(data = update_iterations, index = row_idx),
                upperbound = pd.Series(data = upperbounds, index = row_idx)
                )
        stats['incumbent_update'] = np.where(~np.isnan(stats['iterations']), True, False)
        stats = stats.ffill()

        # add relative gap
        gap = (stats['upperbound'] - stats['lowerbound']) / (np.finfo(float).tiny + stats['upperbound'] )
        stats['gap'] = np.fmin(MAX_GAP, gap)

        # add model ids
        if self.store_solutions:
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
                           'simplex_iterations',
                           'model_ids',
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
# --- end BLM StatsCallback
