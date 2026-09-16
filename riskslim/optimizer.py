"""RiskSLIM optimizer."""

import numpy as np

import cplex
from cplex.exceptions import CplexError
from riskslim.loss_computation import get_loss_functions
from riskslim.utils import Stats, validate_settings, print_log
from riskslim.defaults import DEFAULT_INITIALIZATION_SETTINGS, DEFAULT_LCPA_SETTINGS
from riskslim.mip import add_mip_starts, create_risk_slim, set_cplex_mip_parameters
from riskslim.solution_pool import SolutionPool, FastSolutionPool
from riskslim.heuristics import discrete_descent, sequential_rounding
from riskslim.bounds import Bounds, chained_updates, compute_loss_bounds
from riskslim.warmstart import (
    run_standard_cpa,
    round_solution_pool,
    sequential_round_solution_pool,
    discrete_descent_solution_pool,
    )
from riskslim.callbacks import LossCallback, PolishAndRoundCallback


class RiskSLIMOptimizer:
    """RiskSLIM optimizer object.

    Attributes
    ----------
    X : 2d array
        Observations (rows) and features (columns).
    y : 1d or 2d array
        Class labels.
    coef_set : riskslim.coefficient_set.CoefficientSet
        Constraints on coefficients of input variables
    bounds : riskslim.data.Bounds
        Lower and upper bounds on objective value, loss, and model size.
    stats : riskslim.data.Stats
        Cplex solution statistics.
    solution : cplex SolutionInterface
        Solved cplex solution.
    solution_info : dict
        Additional solution information.
    pool : riskslim.solution_pool.SolutionPool
        Pool of solutions and associated objective values.
    coefficients, rho : 1d array
        Solved cofficients.
    fitted : bool
        Whether model has be fit.
    """

    def __init__(self, data, coef_set, max_size, c0_value=1e-6, max_abs_offset=None, verbose=True, **kwargs):
        """
        Parameters
        ----------
        c0_value : 1d array or float, optional, default: 1e-6
            L0-penalty for all parameters when an integer or for each parameter
            separately when an array.
        max_abs_offset : float, optional, default: None
            Maximum absolute value of intercept. This may be specificed as the first value
            of min_coef and max_coef. However, if min_coef and max_coef are floats, this parameter
            provides a convenient way to set bounds on the offset.
        verbose : bool, optional, default: True
            Prints out log information if True, supresses if False.
        **kwargs
            May include key value pairs:

            - "coef_set" : riskslim.coefficient_set.CoefficientSet
                Contraints (bounds) on coefficients of input variables.
                If None, this is constructed based on values passed to other initalization kwargs.
                If not None, other kwargs may be overwritten.

            - "vtype" : str or list of str
                Variable types for coefficients.
                Must be either "I" for integers or "C" for floats.

            - **settings : unpacked dict
                Settings for warmstart (keys: 'init_*'), cplex (keys: 'cplex_*'), and lattice CPA.
                Defaults are defined in ``defaults.DEFAULT_LCPA_SETTINGS``.

        """
        # Empty fields
        self.fitted = False
        self.has_warmstart = False

        # attach main inputs
        self.data = data
        self.coef_set = coef_set
        assert np.greater(c0_value, 0.0), "c0_value should be positive"
        self.c0_value = c0_value

        # bounds
        self.n_variables = data.d
        self.min_size = 0
        self.max_size = np.minimum(max_size, self.n_variables)

        # coefficient bounds
        self.coef_set.update_intercept_bounds(X = self.data.X, y = self.data.y, max_offset = max_abs_offset)
        self.min_coef = self.coef_set.lb
        self.max_coef = self.coef_set.ub

        # logging
        self.verbose = verbose
        self.log = lambda msg: print_log(msg, print_flag=self.verbose)

        # settings
        kwargs.setdefault('display_cplex_progress', self.verbose)
        settings = validate_settings(kwargs, DEFAULT_LCPA_SETTINGS)
        parsed = {
            'warmstart': {k[5:]: settings[k] for k in settings if k.startswith("init_")},
            'cplex':{k[6:]: settings[k] for k in settings if k.startswith("cplex_")},
            'lcpa':{k: settings[k] for k in settings if settings if not k.startswith(("init_", "cplex_"))}
            }
        self.warmstart_settings = validate_settings(parsed['warmstart'], DEFAULT_INITIALIZATION_SETTINGS)
        self.cplex_settings = parsed['cplex']
        mip_settings = {
            "C_0": self.c0_value,
            "coef_set": self.coef_set,
            "tight_formulation": settings["tight_formulation"],
            "drop_variables": settings["drop_variables"],
            "include_auxillary_variable_for_L0_norm": settings["include_auxillary_variable_for_L0_norm"],
            "include_auxillary_variable_for_objval": settings["include_auxillary_variable_for_objval"],
            }

        # vectorized regularization parameters
        self.L0_reg_ind = self.coef_set.penalized_indices()
        self.C_0 = self.coef_set.c0
        self.C_0[self.L0_reg_ind] = self.c0_value
        self.C_0_nnz = self.C_0[self.L0_reg_ind]

        # loss functions
        handles = get_loss_functions(data, coef_set, loss_computation = settings["loss_computation"], max_size = self.max_size)
        for name, handle in handles.items():
            self.__setattr__(f"compute_{name}", handle)

        # other handles
        self.get_L0_norm = lambda rho: np.count_nonzero(rho[self.L0_reg_ind])
        self.get_L0_penalty = lambda rho: np.sum(self.C_0_nnz * (rho[self.L0_reg_ind] != 0.0))
        self.get_alpha = lambda rho: np.array(abs(rho[self.L0_reg_ind]) > 0.0, dtype=np.float64)
        self.get_L0_penalty_from_alpha = lambda alpha: np.sum(self.C_0_nnz * alpha)
        self.get_objval = lambda rho: self.compute_loss(rho) + np.sum(self.C_0_nnz * (rho[self.L0_reg_ind] != 0.0))

        # set up bounds
        bounds = Bounds(min_size = self.min_size, max_size=self.max_size)
        bounds.loss_min, bounds.loss_max = compute_loss_bounds(self.data, self.coef_set, self.max_size)

        # solution pool
        pool = SolutionPool(self.n_variables)

        # Check if trivial solution is feasible, if so add it to the pool and update bounds
        trivial_solution = np.zeros(self.n_variables, dtype=np.float64)
        if self.is_feasible(trivial_solution):
            trivial_objval = self.compute_loss(trivial_solution)
            if settings["initial_bound_updates"]:
                bounds.objval_max = min(bounds.objval_max, trivial_objval)
                bounds.loss_max = min(bounds.loss_max, trivial_objval)
                bounds = chained_updates(bounds, self.C_0_nnz)
            pool.add(objvals=trivial_objval, solutions=trivial_solution)

        # Attach
        self.pool = pool
        self.bounds = bounds
        self.cut_queue = FastSolutionPool(self.n_variables)
        self.polish_queue = FastSolutionPool(self.n_variables)
        self.stats = Stats(incumbent = np.full(self.n_variables, np.nan), bounds=self.bounds)

        # warmstart procedure
        initial_cuts = None
        if settings["initialization_flag"] :
            initial_cuts, pool, bounds = self.warmstart(self.pool, self.bounds, mip_settings, **parsed['warmstart'])
            self.pool.append(pool)
            self.bounds = bounds


        # create riskslim mip
        mip_settings.update(bounds.asdict())
        cpx, indices = create_risk_slim(coef_set=self.coef_set, settings= mip_settings)
        indices.update({"C_0_nnz": self.C_0_nnz, "L0_reg_ind": self.L0_reg_ind})
        self.mip_indices = indices

        # add constraints
        # todo: remove this and find way to access cpx from RiskSLIM directly
        # cons = cpx.linear_constraints
        # for name, var_inds, values, sense, rhs in self.constraints:
        #     name = f"con_{cons.get_num()}" if name is None else name
        #     cons.add(names=[name], lin_expr=[cplex.SparsePair(ind=var_inds, val=values)], senses=[sense],rhs=[rhs])

        # todo: remove this
        loss_cb = cpx.register_callback(LossCallback)
        loss_cb.initialize(indices=indices,
                           stats=self.stats,
                           settings=settings,
                           compute_loss_cut=self.compute_loss_cut,
                           get_alpha=self.get_alpha,
                           get_L0_penalty_from_alpha=self.get_L0_penalty_from_alpha,
                           initial_cuts=initial_cuts,
                           cut_queue=self.cut_queue,
                           polish_queue=self.polish_queue,
                           verbose=self.verbose,
                           )

        # add heuristic callback if rounding or polishing
        heuristic_cb = None
        if settings["round_flag"] or settings["polish_flag"]:
            heuristic_cb = cpx.register_callback(PolishAndRoundCallback)
            active_set_flag = self.max_size <= self.n_variables
            polisher = lambda rho: discrete_descent(
                    rho,
                    self.data.Z,
                    self.C_0,
                    self.coef_set.ub,
                    self.coef_set.lb,
                    self.get_L0_penalty,
                    self.compute_loss_from_scores,
                    active_set_flag,
                    )

            rounder = lambda rho, cutoff: sequential_rounding(
                    rho,
                    self.data.Z,
                    self.C_0,
                    self.compute_loss_from_scores_real,
                    self.get_L0_penalty,
                    cutoff,
                    )

            heuristic_cb.initialize(
                    indices=indices,
                    control=self.stats,
                    settings=settings,
                    cut_queue=self.cut_queue,
                    polish_queue=self.polish_queue,
                    get_objval=self.get_objval,
                    get_L0_norm=self.get_L0_norm,
                    is_feasible=self.is_feasible,
                    polishing_handle=polisher,
                    rounding_handle=rounder,
                    )

            self.heuristic_callback = heuristic_cb

        # initialize solution pool
        if len(self.pool) > 0:
            if settings["polish_flag"]:
                self.polish_queue.add(self.pool.objvals[0], self.pool.solutions[0])
            else:
                cpx = add_mip_starts(cpx, self.mip_indices, self.pool, mip_start_effort_level=cpx.MIP_starts.effort_level.repair)
            if settings["add_cuts_at_heuristic_solutions"] and len(self.pool) > 1:
                self.cut_queue.add(self.pool.objvals[1:], self.pool.solutions[1:])

        # finalize
        self.mip = set_cplex_mip_parameters(cpx, parsed['cplex'], display_cplex_progress=settings["display_cplex_progress"])
        self.mip_settings = mip_settings
        self.loss_callback = loss_cb
        self.heuristic_cb = heuristic_cb
        self.settings = settings

    def warmstart(self, pool, bounds, mip_settings, **kwargs):
        """Run an initialization routine to speed up RiskSLIM.

        Parameters
        ----------
        warmstart_settings : dict
            Warmstart settings to overwrite settings passed on initalization.
        """
        settings = validate_settings(kwargs, self.warmstart_settings)
        settings["type"] = "cvx"

        # Construct LP relaxation
        lp_settings = dict(mip_settings)
        lp_settings["relax_integer_variables"] = True
        cpx, cpx_indices = create_risk_slim(coef_set=self.coef_set, settings=lp_settings)
        cpx = set_cplex_mip_parameters(cpx, self.cplex_settings, display_cplex_progress=settings["display_cplex_progress"])

        # Solve RiskSLIM LP using standard CPA
        stats, cuts, pool = run_standard_cpa(
                cpx=cpx,
                cpx_indices=cpx_indices,
                compute_loss=self.compute_loss_real,
                compute_loss_cut=self.compute_loss_cut_real,
                settings=settings,
                print_flag=self.verbose,
                )

        # Update cuts
        self.log(f"warmstart CPA produced {len(cuts['coefs'])} cuts")

        # Update bounds
        bounds = chained_updates(bounds, self.C_0_nnz, new_objval_at_relaxation=stats["lowerbound"])

        constraints = {
            "coef_set": self.coef_set,
            "min_size": self.min_size,
            "max_size": self.max_size,
            }

        # todo move into factory function
        def rounded_model_size_is_ok(rho):
            zero_idx_rho_ceil = np.equal(np.ceil(rho), 0)
            zero_idx_rho_floor = np.equal(np.floor(rho), 0)
            cannot_round_to_zero = np.logical_not(np.logical_or(zero_idx_rho_ceil, zero_idx_rho_floor))
            rounded_min_coef_size = np.count_nonzero(cannot_round_to_zero[self.L0_reg_ind])
            rounded_max_coef_size = np.count_nonzero(rho[self.L0_reg_ind])
            return (
                    rounded_min_coef_size >= self.min_size >= 0
                    and rounded_max_coef_size <= self.max_size
            )

        pool = pool.remove_infeasible(rounded_model_size_is_ok).distinct().sort()

        if len(pool) == 0:
            self.log("all CPA solutions are infeasible")

        # Round CPA solutions
        if settings["use_rounding"] and len(pool) > 0:
            self.log(f"running naive rounding on {len(pool)} solutions")
            self.log("best objective value: {:04f}".format(np.min(pool.objvals)))
            rnd_pool, _, _ = round_solution_pool(
                    pool,
                    constraints,
                    max_runtime=settings["rounding_max_runtime"],
                    max_solutions=settings["rounding_max_solutions"],
                    )
            rnd_pool = rnd_pool.compute_objvals(self.get_objval).remove_infeasible(self.is_feasible)
            self.log(f"rounding produced {len(rnd_pool)} integer solutions")
            if len(rnd_pool) > 0:
                pool.append(rnd_pool)
            self.log("best objective value: {:04f}".format(np.min(pool.objvals)))

        # Sequentially Round CPA solutions
        if settings["use_sequential_rounding"] and len(pool) > 0:
            self.log(f"running sequential rounding on {len(pool)} solutions")
            self.log("best objective value: {:04f}".format(np.min(pool.objvals)))
            sqrnd_pool, _, _ = sequential_round_solution_pool(
                    pool=pool,
                    Z=self.data.Z,
                    C_0=self.C_0,
                    compute_loss_from_scores_real=self.compute_loss_from_scores_real,
                    get_L0_penalty=self.get_L0_penalty,
                    max_runtime=settings["sequential_rounding_max_runtime"],
                    max_solutions=settings["sequential_rounding_max_solutions"],
                    objval_cutoff=bounds.objval_max,
                    )

            sqrnd_pool = sqrnd_pool.remove_infeasible(self.is_feasible)
            self.log(f"sequential rounding produced {len(sqrnd_pool)} integer solutions")

            if len(sqrnd_pool) > 0:
                pool = pool.append(sqrnd_pool)
                self.log("best objective value: {:04f}".format(np.min(pool.objvals)))

        # Polish rounded solutions
        if settings["polishing_after"] and len(pool) > 0:
            self.log(f"polishing {len(pool)} solutions")
            self.log("best objective value: %1.4f" % np.min(pool.objvals))
            dcd_pool, _, _ = discrete_descent_solution_pool(
                    pool=pool,
                    Z=self.data.Z,
                    C_0=self.C_0,
                    constraints=constraints,
                    compute_loss_from_scores=self.compute_loss_from_scores,
                    get_L0_penalty=self.get_L0_penalty,
                    max_runtime=settings["polishing_max_runtime"],
                    max_solutions=settings["polishing_max_solutions"],
                    )
            dcd_pool = dcd_pool.remove_infeasible(self.is_feasible)
            if len(dcd_pool) > 0:
                self.log(f"polishing produced {len(dcd_pool)} integer solutions")
                pool.append(dcd_pool)

        # Remove solutions that are not feasible, not integer
        if len(pool) > 0:
            pool = pool.remove_nonintegral().distinct().sort()

        # Update upper and lower bounds
        self.log(f"initialization produced {len(pool)} feasible solutions")

        if len(pool) > 0:
            bounds = chained_updates(bounds, self.C_0_nnz, new_objval_at_feasible=np.min(pool.objvals))
            self.log("best objective value: {:04f}".format(np.min(pool.objvals)))

        return cuts, pool, bounds

    def optimize(self, X, y, sample_weights=None, max_runtime = 60.0):
        """Optimize RiskSLIM.

        Parameters
        ----------
        X : 2d-array
            Observations (rows) and features (columns).
            With an addtional column of 1s for the intercept.
        y : 2d-array
            Class labels (+1, -1) with shape (n_rows, 1).
        sample_weights : 2d array, optional, default: None
            Sample weights with shape (n_features, 1). Must all be positive.
        """

        # Set cplex parameters and runtime
        self.mip.parameters.timelimit.set(self.settings["max_runtime"])
        self.mip.solve()
        self.fitted = True

    def add_constraint(self, var_names, var_type, values, rhs, sense, name=None):
        """Add a constraint to the MIP.

        Parameters
        ----------
        var_names : list of string
            Variable names to use in constraint.
        var_type : {"rho", "alpha"}
            Either the coefficient (rho) or use term (alpha).
            Alpha is in {0, 1}. Rho is the coefficient being learned.
        values : list of float
            Coefficients to multiply the varible by.
        rhs : float
            Right hand side of the inequality.
        sense : {"G", "L", "E", "R"}
            Must be one of 'G', 'L', 'E', and 'R', indicating greater-than,
            less-than, equality, and ranged constraints, respectively.
        name : str, optional, default: None
            Name of the constraint being added.
            For accessibility from the cplex mip object.

        Notes
        -----
        Formulated as:

            values[0]*variable[0] + ... + values[i]*variable[i] (<, <=, >, >=) rhs,
            where variable[i] is either the coefficient (rho) or use term (alpha).

        """

        # Checks
        if var_type not in ("rho", "alpha"):
            raise ValueError("var_type must be in [\"rho\", \"alpha\"]")

        if len(var_names) != len(values):
            raise ValueError("var_name and values must be the same length.")

        # Anon funcs to get name and index set in mip object (e.g. rho_i, alpha_i)
        get_name = lambda var_name: var_type + '_' + str(self._data.variable_names.index(var_name))
        get_ind = lambda var_names: [get_name(v) for v in var_names]

        # Add constraint to a queue - it is added to the mip solver add fit time
        self.constraints.append([name, get_ind(var_names), values, sense, rhs])

    #### properties ####
    @property
    def solution(self):
        """Returns CPLEX solution.

        Returns
        -------
        cplex SolutionInterface
        """
        # todo add wrapper if solution does not exist
        return self.mip.solution

    @property
    def coefficients(self):
        """
        Returns
        -------
        coefs : np.ndarray
            C
            oefficients of the linear classifier
        """
        s = self.solution
        if s.is_primal_feasible():
            coefs = np.array(s.get_values(self.mip_indices["rho"]))
        else:
            coefs = np.repeat(np.nan, self.n_variables)
        return coefs

    @property
    def solution_info(self):
        """Returns information associated with the current best solution for the mip.

        Returns
        -------
        info : dict
            Contains best solution info.
        """
        # Record mip solution statistics
        solution = self.solution

        try:
            self.stats.incumbent = np.array(
                    solution.get_values(self.mip_indices["rho"])
                    )
            self.stats.upperbound = solution.get_objective_value()
            self.stats.lowerbound = solution.MIP.get_best_objective()
            self.stats.relative_gap = solution.MIP.get_mip_relative_gap()
            self.stats.found_solution = True
        except CplexError:
            self.stats.found_solution = False

        self.stats.cplex_status = solution.get_status_string()
        self.stats.total_callback_time = (
                self.stats.total_cut_callback_time
                + self.stats.total_heuristic_callback_time
        )
        self.stats.total_solver_time = (
                self.stats.total_run_time - self.stats.total_callback_time
        )
        self.stats.total_data_time = (
                self.stats.total_cut_time
                + self.stats.total_polish_time
                + self.stats.total_round_time
                + self.stats.total_round_then_polish_time
        )

        # Output for Model
        info = {
            "c0_value": self.c0_value,
            "solution": self.stats.incumbent,
            "objective_value": float("inf"),
            "loss_value":  float("inf"),
            "optimality_gap": float("inf"),
            "run_time": self.stats.total_run_time,
            "solver_time": self.stats.total_solver_time,
            "callback_time": self.stats.total_callback_time,
            "data_time": self.stats.total_data_time,
            "nodes_processed": self.stats.nodes_processed,
            }

        if self.stats.found_solution:
            info.update({
                "objective_value": self.get_objval(self.stats.incumbent),
                "loss_value": self.compute_loss(self.stats.incumbent),
                "optimality_gap": self.stats.relative_gap
                })

        return info

    # helper functions
    def is_feasible(self, rho):
        """Ensure constraints are obeyed.

        Parameters
        ----------
        """
        flag = np.greater_equal(self.coef_set.ub, rho).all() and np.less_equal(self.coef_set.lb, rho).all()
        flag = (self.min_size <= np.count_nonzero(rho[self.L0_reg_ind]) <= self.max_size)
        return flag
