"""CPLEX backend of the RiskSLIM MIP."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from cplex import Cplex, SparsePair
from cplex import infinity as CPX_INFINITY

from riskslim.coefficient_set import CoefficientSet
from riskslim.defaults import DEFAULT_CPLEX_SETTINGS
from riskslim.opt.cpx.callbacks import LossCallback, PolishAndRoundCallback
from riskslim.opt.cpx.utils import add_mip_start, set_cpx_display_options
from riskslim.opt.mip import (
    L0_NORM_DEF_NAME,
    L0_NORM_NAME,
    LOSS_NAME,
    OBJVAL_DEF_NAME,
    OBJVAL_NAME,
    L0_norm_lb_name,
    L0_norm_ub_name,
    RiskSLIMMIP,
    alpha_name,
    build_mip_indices,
    cast_mip_start,
    convert_to_risk_slim_solution,
    rho_name,
)
from riskslim.utils import print_log

if TYPE_CHECKING:
    from riskslim.solution_pool import FastSolutionPool, SolutionPool
    from riskslim.utils import Stats

OPTIMAL_STATUS_NAMES = ("optimal", "optimal_tolerance", "MIP_optimal")


class CplexRiskSLIMMIP(RiskSLIMMIP):
    """RiskSLIM surrogate MIP on CPLEX.

    Attributes:
        cpx: The native ``cplex.Cplex`` model; ``None`` before ``build``.
        indices: Indices of the built MIP (see ``riskslim.opt.mip.build_mip_indices``).
    """

    def __init__(self) -> None:
        """Create an empty problem; call ``build`` next."""
        super().__init__()
        self.cpx: Cplex | None = None

    # ---- build and configure

    def build(self, coef_set: CoefficientSet, settings: dict) -> dict:
        """Build the RiskSLIM surrogate MIP (no loss cuts yet) as a ``cplex.Cplex``.

        Formulation::

            minimize    loss + sum_j C_0j * alpha_j
            such that   rho_lb_j * alpha_j <= rho_j <= rho_ub_j * alpha_j
                        min_size <= L0_norm = sum_j alpha_j <= max_size
                        objval_min <= objval = loss + sum_j C_0j * alpha_j <= objval_max
                        rho_j in [rho_lb_j, rho_ub_j], alpha_j in {0, 1}

        ``objval`` and ``L0_norm`` are auxiliary variables (needed by the callbacks); each is
        added when requested or when its bounds are non-trivial. With ``drop_variables``, the
        L0 constraints of sign-constrained coefficients and the ``alpha_j`` of fixed
        coefficients are dropped; the ``alpha_j`` and L0 constraints of ``(Intercept)`` are
        always dropped.

        Args:
            coef_set: Bounds, types, and penalties of the coefficients.
            settings: MIP formulation settings; defaults are filled in place.

        Returns:
            ``self.indices``.
        """
        assert isinstance(coef_set, CoefficientSet)
        assert isinstance(settings, dict)

        log = (
            (lambda msg: print_log(msg))
            if settings.get("print_flag", False)
            else (lambda msg: None)
        )

        # set default parameters
        settings.setdefault("C_0", 0.01)
        settings.setdefault("w_pos", 1.0)
        settings.setdefault("w_neg", 2.0 - settings["w_pos"])
        settings.setdefault("include_auxillary_variable_for_objval", True)
        settings.setdefault("include_auxillary_variable_for_L0_norm", True)
        settings.setdefault("loss_min", 0.00)
        settings.setdefault("loss_max", float(CPX_INFINITY))
        settings.setdefault("min_size", 0)
        settings.setdefault("max_size", len(coef_set))
        settings.setdefault("objval_min", 0.00)
        settings.setdefault("objval_max", float(CPX_INFINITY))
        settings.setdefault("relax_integer_variables", False)
        settings.setdefault("drop_variables", True)
        settings.setdefault("tight_formulation", False)
        settings.setdefault("set_cplex_cutoffs", True)

        # variables
        P = len(coef_set)
        w_pos = settings["w_pos"]
        C_0j = np.copy(coef_set.c0)
        L0_reg_ind = np.isnan(C_0j) + C_0j != 0.0
        C_0j[L0_reg_ind] = (
            settings["C_0"][L0_reg_ind]
            if isinstance(settings["C_0"], np.ndarray)
            else settings["C_0"]
        )
        C_0j = C_0j.tolist()
        C_0_rho = np.copy(C_0j)
        trivial_min_size = 0
        trivial_max_size = np.sum(L0_reg_ind)

        rho_ub = list(coef_set.ub)
        rho_lb = list(coef_set.lb)
        rho_type = "".join(list(coef_set.vtype))

        # min/max values for loss
        loss_min = max(0.0, float(settings["loss_min"]))
        loss_max = min(CPX_INFINITY, float(settings["loss_max"]))

        # min/max values for model size
        min_size = np.maximum(settings["min_size"], 0.0)
        max_size = np.minimum(settings["max_size"], trivial_max_size)
        min_size = np.ceil(min_size)
        max_size = np.floor(max_size)
        assert min_size <= max_size

        # min/max values for objval
        objval_min = max(settings["objval_min"], 0.0)
        objval_max = min(settings["objval_max"], CPX_INFINITY)
        assert objval_min <= objval_max

        # include auxiliary variable for model size?
        nontrivial_min_size = min_size > trivial_min_size
        nontrivial_max_size = max_size < trivial_max_size
        include_L0_norm = (
            settings["include_auxillary_variable_for_L0_norm"]
            or nontrivial_min_size
            or nontrivial_max_size
        )

        # include auxiliary variable for objective value?
        nontrivial_objval_min = objval_min > 0.0
        nontrivial_objval_max = objval_max < CPX_INFINITY
        include_objval = (
            settings["include_auxillary_variable_for_objval"]
            or nontrivial_objval_min
            or nontrivial_objval_max
        )

        has_intercept = "(Intercept)" in coef_set.variable_names

        # create MIP object
        cpx = Cplex()
        variables = cpx.variables
        cons = cpx.linear_constraints
        cpx.objective.set_sense(cpx.objective.sense.minimize)

        # main variables: x = [loss, rho_j, alpha_j]
        loss_obj = [w_pos]
        loss_names = [LOSS_NAME]
        rho_names = [rho_name(j) for j in range(P)]
        alpha_names = [alpha_name(j) for j in range(P)]

        obj = loss_obj + [0.0] * P + C_0j
        ub = [loss_max] + rho_ub + [1.0] * P
        lb = [loss_min] + rho_lb + [0.0] * P
        ctype = "C" + rho_type + "B" * P
        varnames = loss_names + rho_names + alpha_names

        if include_objval:
            log(
                "adding auxiliary variable for objval s.t. %1.4f <= objval <= %1.4f"
                % (objval_min, objval_max)
            )
            obj += [0.0]
            ub += [objval_max]
            lb += [objval_min]
            varnames += [OBJVAL_NAME]
            ctype += "C"

        if include_L0_norm:
            log(
                "adding auxiliary variable for L0_norm s.t. %d <= L0_norm <= %d"
                % (min_size, max_size)
            )
            obj += [0.0]
            ub += [float(max_size)]
            lb += [float(min_size)]
            varnames += [L0_NORM_NAME]
            ctype += "I"

        if settings["relax_integer_variables"]:
            ctype = ctype.replace("I", "C")
            ctype = ctype.replace("B", "C")

        variables.add(obj=obj, lb=lb, ub=ub, types=ctype, names=varnames)

        # L0-norm LB constraints: 0 <= rho_j - rho_lb_j * alpha_j
        for j in range(P):
            cons.add(
                names=[L0_norm_lb_name(j)],
                lin_expr=[SparsePair(ind=[rho_names[j], alpha_names[j]], val=[1.0, -rho_lb[j]])],
                senses="G",
                rhs=[0.0],
            )

        # L0-norm UB constraints: 0 <= -rho_j + rho_ub_j * alpha_j
        for j in range(P):
            cons.add(
                names=[L0_norm_ub_name(j)],
                lin_expr=[SparsePair(ind=[rho_names[j], alpha_names[j]], val=[-1.0, rho_ub[j]])],
                senses="G",
                rhs=[0.0],
            )

        # objval definition: objval = loss + sum(C_0j * alpha_j)
        if include_objval:
            log("adding constraint so that objective value <= " + str(objval_max))
            cons.add(
                names=[OBJVAL_DEF_NAME],
                lin_expr=[
                    SparsePair(
                        ind=[OBJVAL_NAME] + loss_names + alpha_names, val=[-1.0] + loss_obj + C_0j
                    )
                ],
                senses="E",
                rhs=[0.0],
            )

        # L0_norm definition: L0_norm = sum(alpha_j)
        if include_L0_norm:
            cons.add(
                names=[L0_NORM_DEF_NAME],
                lin_expr=[SparsePair(ind=[L0_NORM_NAME] + alpha_names, val=[1.0] + [-1.0] * P)],
                senses="E",
                rhs=[0.0],
            )

        dropped_variables = []
        constraints_to_drop = []

        if settings["drop_variables"]:
            # drop L0_norm_lb/ub constraints for coefficients with rho_lb >= 0 / rho_ub <= 0
            sign_pos_ind = np.flatnonzero(coef_set.sign > 0)
            sign_neg_ind = np.flatnonzero(coef_set.sign < 0)
            constraints_to_drop.extend([L0_norm_lb_name(j) for j in sign_pos_ind])
            constraints_to_drop.extend([L0_norm_ub_name(j) for j in sign_neg_ind])

            # drop alpha for coefficients with rho_ub == rho_lb
            fixed_value_ind = np.flatnonzero(coef_set.ub == coef_set.lb)
            variables_to_drop = [alpha_name(j) for j in fixed_value_ind]
            variables.delete(variables_to_drop)
            dropped_variables += variables_to_drop
            alpha_names = [
                alpha_names[j] for j in range(P) if alpha_names[j] not in dropped_variables
            ]

        # drop alpha / L0_norm_ub / L0_norm_lb for '(Intercept)'
        if has_intercept:
            intercept_idx = coef_set.variable_names.index("(Intercept)")
            intercept_alpha_name = alpha_name(intercept_idx)

            # intercept alpha may already be dropped above when rho_ub == rho_lb
            if intercept_alpha_name not in dropped_variables:
                variables.delete([intercept_alpha_name])
                alpha_names.remove(intercept_alpha_name)
                dropped_variables.append(intercept_alpha_name)

            log("dropped L0 indicator for '(Intercept)'")
            constraints_to_drop.extend(
                [L0_norm_ub_name(intercept_idx), L0_norm_lb_name(intercept_idx)]
            )

        if len(constraints_to_drop) > 0:
            constraints_to_drop = list(set(constraints_to_drop))
            cons.delete(constraints_to_drop)

        indices = build_mip_indices(
            variable_names=variables.get_names(),
            n_constraints=cons.get_num(),
            rho_names=rho_names,
            alpha_names=alpha_names,
            L0_reg_ind=L0_reg_ind,
            C_0_rho=C_0_rho,
            C_0_alpha=cpx.objective.get_linear(alpha_names) if len(alpha_names) > 0 else [],
            include_objval=include_objval,
            include_L0_norm=include_L0_norm,
        )

        # officially change the problem to LP if variables are relaxed
        if settings["relax_integer_variables"]:
            old_problem_type = cpx.problem_type[cpx.get_problem_type()]
            cpx.set_problem_type(cpx.problem_type.LP)
            new_problem_type = cpx.problem_type[cpx.get_problem_type()]
            log("changed problem type from %s to %s" % (old_problem_type, new_problem_type))

        if settings["set_cplex_cutoffs"] and not settings["relax_integer_variables"]:
            cpx.parameters.mip.tolerances.lowercutoff.set(objval_min)
            cpx.parameters.mip.tolerances.uppercutoff.set(objval_max)

        self.cpx = cpx
        self.indices = indices
        return indices

    def set_parameters(self, settings: dict, display_progress: bool = False) -> None:
        """Set CPLEX parameters; MIP-only parameters apply only when the problem is a MILP.

        Output is silenced when ``display_progress`` is ``None`` or ``False``.

        Args:
            settings: ``cplex_*`` settings with the prefix removed.
            display_progress: Show CPLEX progress output when True.
        """
        cpx = self.cpx
        p = cpx.parameters
        p.randomseed.set(settings["randomseed"])
        p.threads.set(settings["n_cores"])
        p.output.clonelog.set(0)
        p.parallel.set(1)

        if display_progress in (None, False):
            set_cpx_display_options(
                cpx, display_mip=False, display_lp=False, display_parameters=False
            )

        p.simplex.tolerances.optimality.set(
            settings.get("optimality_tolerance", DEFAULT_CPLEX_SETTINGS["optimality_tolerance"])
        )

        if cpx.problem_type[cpx.get_problem_type()] == "MILP":
            # MIP parameters
            p.emphasis.mip.set(settings["mipemphasis"])
            p.mip.tolerances.mipgap.set(settings["mipgap"])
            p.mip.tolerances.absmipgap.set(settings["absmipgap"])
            p.mip.tolerances.integrality.set(settings["integrality_tolerance"])

            # solution pool parameters
            p.mip.limits.repairtries.set(settings["repairtries"])
            p.mip.pool.capacity.set(settings["poolsize"])
            # 0 = replace oldest / 1 = replace worst objective / 2 = replace least diverse
            p.mip.pool.replace.set(settings["poolreplace"])

    def set_time_limit(self, seconds: float) -> None:
        """Limit the wall-clock time of the next ``solve``.

        Args:
            seconds: Time limit in seconds.
        """
        self.cpx.parameters.timelimit.set(seconds)

    def get_variable_bounds(self, idx: int) -> tuple[float, float]:
        """Return ``(lb, ub)`` of variable ``idx``.

        Args:
            idx: Variable index.

        Returns:
            Lower and upper bound.
        """
        variables = self.cpx.variables
        return variables.get_lower_bounds(idx), variables.get_upper_bounds(idx)

    def set_variable_bounds(self, idx: int, lb: float, ub: float) -> None:
        """Set the bounds of variable ``idx`` (lower bound first, then upper bound).

        Args:
            idx: Variable index.
            lb: New lower bound.
            ub: New upper bound.
        """
        variables = self.cpx.variables
        variables.set_lower_bounds(idx, lb)
        variables.set_upper_bounds(idx, ub)

    def add_cut(self, idx: Sequence[int], coefs: Sequence[float], rhs: float) -> None:
        """Add the constraint ``sum_k coefs[k] * x[idx[k]] >= rhs``.

        Args:
            idx: Variable indices.
            coefs: Coefficients, aligned with ``idx``.
            rhs: Right-hand side.
        """
        self.cpx.linear_constraints.add(
            lin_expr=[SparsePair(ind=idx, val=coefs)], senses=["G"], rhs=[rhs]
        )

    def get_cuts(self) -> dict:
        """Return every constraint added after ``build``, in order.

        Returns:
            ``{"coefs": [SparsePair, ...], "lhs": [rhs, ...]}``.
        """
        cons = self.cpx.linear_constraints
        idx = list(range(self.indices["n_constraints"], cons.get_num(), 1))
        return {"coefs": cons.get_rows(idx), "lhs": cons.get_rhs(idx)}

    def add_mip_starts(self, pool: SolutionPool, max_mip_starts: float = float("inf")) -> None:
        """Add the distinct solutions of ``pool`` as MIP starts (effort level ``repair``).

        Args:
            pool: Pool of RiskSLIM coefficient vectors and objective values.
            max_mip_starts: Maximum number of starts to add.
        """
        cpx = self.cpx
        effort_level = cpx.MIP_starts.effort_level.repair

        try:
            obj_cutoff = cpx.parameters.mip.tolerances.uppercutoff.get()
        except Exception:
            obj_cutoff = float("inf")

        variable_types = cpx.variables.get_types(list(range(self.indices["n_variables"])))
        pool = pool.distinct().sort()

        n_added = 0
        for objval, rho in zip(pool.objvals, pool.solutions):
            if np.less_equal(objval, obj_cutoff):
                values, _ = convert_to_risk_slim_solution(
                    rho=rho, indices=self.indices, objval=objval
                )
                values = cast_mip_start(values, variable_types)
                add_mip_start(cpx, values, effort_level=effort_level, name=f"mip_start_{n_added}")
                n_added += 1

            if n_added >= max_mip_starts:
                break

    def register_callbacks(
        self,
        *,
        stats: Stats,
        settings: dict,
        compute_loss_cut: Callable,
        get_alpha: Callable,
        get_L0_penalty_from_alpha: Callable,
        cut_queue: FastSolutionPool,
        polish_queue: FastSolutionPool,
        initial_cuts: dict | None = None,
        get_objval: Callable | None = None,
        get_L0_norm: Callable | None = None,
        is_feasible: Callable | None = None,
        polishing_handle: Callable | None = None,
        rounding_handle: Callable | None = None,
        verbose: bool = True,
    ) -> tuple[LossCallback, PolishAndRoundCallback | None]:
        """Register ``LossCallback`` and, if rounding or polishing, ``PolishAndRoundCallback``.

        See ``RiskSLIMMIP.register_callbacks`` for the arguments.

        Returns:
            ``(loss_callback, heuristic_callback)``; ``heuristic_callback`` is ``None`` unless
            ``settings["round_flag"]`` or ``settings["polish_flag"]``.
        """
        loss_cb = self.cpx.register_callback(LossCallback)
        loss_cb.initialize(
            indices=self.indices,
            stats=stats,
            settings=settings,
            compute_loss_cut=compute_loss_cut,
            get_alpha=get_alpha,
            get_L0_penalty_from_alpha=get_L0_penalty_from_alpha,
            initial_cuts=initial_cuts,
            cut_queue=cut_queue,
            polish_queue=polish_queue,
            verbose=verbose,
        )

        heuristic_cb = None
        if settings["round_flag"] or settings["polish_flag"]:
            heuristic_cb = self.cpx.register_callback(PolishAndRoundCallback)
            heuristic_cb.initialize(
                indices=self.indices,
                control=stats,
                settings=settings,
                cut_queue=cut_queue,
                polish_queue=polish_queue,
                get_objval=get_objval,
                get_L0_norm=get_L0_norm,
                is_feasible=is_feasible,
                polishing_handle=polishing_handle,
                rounding_handle=rounding_handle,
            )

        return loss_cb, heuristic_cb

    # ---- solve and read the solution

    def solve(self) -> None:
        """Solve the model in place."""
        self.cpx.solve()

    def has_solution(self) -> bool:
        """Return True if the last solve found a feasible solution."""
        return bool(self.cpx.solution.is_primal_feasible())

    def is_optimal(self) -> bool:
        """Return True if the last solve ended optimal (within tolerance)."""
        return self.status_name() in OPTIMAL_STATUS_NAMES

    def status_name(self) -> str:
        """Return the CPLEX status name of the last solve (e.g. ``"MIP_optimal"``)."""
        solution = self.cpx.solution
        return solution.status[solution.get_status()]

    def status_string(self) -> str:
        """Return the CPLEX status string of the last solve."""
        return self.cpx.solution.get_status_string()

    def get_values(self, idx: Sequence[int]) -> np.ndarray:
        """Return the values of variables ``idx`` in the best solution.

        Args:
            idx: Variable indices.

        Returns:
            1d array of values, aligned with ``idx``.
        """
        return np.array(self.cpx.solution.get_values(idx))

    def objective_value(self) -> float:
        """Return the objective value of the best solution."""
        return self.cpx.solution.get_objective_value()

    def best_bound(self) -> float:
        """Return the best lower bound on the MIP objective."""
        return self.cpx.solution.MIP.get_best_objective()

    def relative_gap(self) -> float:
        """Return CPLEX's relative MIP gap."""
        return self.cpx.solution.MIP.get_mip_relative_gap()

    def simplex_iteration_count(self) -> int:
        """Return the number of simplex iterations of the last solve."""
        return int(self.cpx.solution.progress.get_num_iterations())
