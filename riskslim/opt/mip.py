"""Solver-agnostic RiskSLIM MIP: the abstract problem, naming scheme, and solution helpers.

``RiskSLIMMIP`` is the contract every solver backend implements. Its method set is exactly
the set of operations that ``riskslim/optimizer.py`` and ``riskslim/warmstart.py`` perform on
the CPLEX model and its solution; the CPLEX *callback* API used by
``riskslim/opt/cpx/callbacks.py`` stays inside each backend's callback classes, which
``register_callbacks`` attaches.

Each method docstring records how a CPLEX subclass and a SCIP (PySCIPOpt) subclass would
satisfy it. Only the CPLEX backend exists (``riskslim.opt.cpx.solver``); the SCIP notes are a
design check.

This module does not import ``cplex``; ``load`` imports the backend lazily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from riskslim.utils import check_cplex

if TYPE_CHECKING:
    from riskslim.coefficient_set import CoefficientSet
    from riskslim.solution_pool import FastSolutionPool, SolutionPool
    from riskslim.utils import Stats

# ---- variable and constraint naming scheme

LOSS_NAME = "loss"
OBJVAL_NAME = "objval"
L0_NORM_NAME = "L0_norm"
OBJVAL_DEF_NAME = "objval_def"
L0_NORM_DEF_NAME = "L0_norm_def"


def rho_name(j: int) -> str:
    """Return the MIP variable name of coefficient ``j`` (``rho_j``)."""
    return f"rho_{j}"


def alpha_name(j: int) -> str:
    """Return the MIP variable name of the L0 indicator of coefficient ``j`` (``alpha_j``)."""
    return f"alpha_{j}"


def L0_norm_lb_name(j: int) -> str:
    """Return the name of the constraint ``rho_j >= rho_lb_j * alpha_j``."""
    return f"L0_norm_lb_{j}"


def L0_norm_ub_name(j: int) -> str:
    """Return the name of the constraint ``rho_j <= rho_ub_j * alpha_j``."""
    return f"L0_norm_ub_{j}"


# ---- solver-agnostic helpers


def build_mip_indices(
    variable_names: Sequence[str],
    n_constraints: int,
    rho_names: Sequence[str],
    alpha_names: Sequence[str],
    L0_reg_ind: np.ndarray,
    C_0_rho: np.ndarray,
    C_0_alpha: Sequence[float],
    include_objval: bool,
    include_L0_norm: bool,
) -> dict:
    """Build the ``indices`` dict of a built RiskSLIM MIP.

    Variable indices are positions in ``variable_names``, which must list every variable of
    the built model in solver order (after any variables were dropped).

    Args:
        variable_names: Names of all model variables, in solver index order.
        n_constraints: Number of linear constraints in the built model (before any cuts).
        rho_names: Names of the coefficient variables, one per coefficient.
        alpha_names: Names of the L0 indicator variables kept in the model.
        L0_reg_ind: Boolean mask of the coefficients that are L0-penalized.
        C_0_rho: Per-coefficient L0 penalties.
        C_0_alpha: Objective coefficients of the ``alpha_names`` variables.
        include_objval: Whether the model has the auxiliary ``objval`` variable.
        include_L0_norm: Whether the model has the auxiliary ``L0_norm`` variable.

    Returns:
        Dict with keys ``n_variables``, ``n_constraints``, ``names``, ``loss_names``,
        ``rho_names``, ``alpha_names``, ``loss``, ``rho``, ``alpha``, ``L0_reg_ind``,
        ``C_0_rho``, ``C_0_alpha``, plus ``objval_name``/``objval`` and
        ``L0_norm_name``/``L0_norm`` when those variables exist. ``loss``, ``rho``, and
        ``alpha`` are lists of indices; ``objval`` and ``L0_norm`` are single indices.
    """
    names = list(variable_names)
    position = {name: k for k, name in enumerate(names)}
    loss_names = [LOSS_NAME]
    rho_names = list(rho_names)
    alpha_names = list(alpha_names)

    indices = {
        "n_variables": len(names),
        "n_constraints": n_constraints,
        "names": names,
        "loss_names": loss_names,
        "rho_names": rho_names,
        "alpha_names": alpha_names,
        "loss": [position[n] for n in loss_names],
        "rho": [position[n] for n in rho_names],
        "alpha": [position[n] for n in alpha_names],
        "L0_reg_ind": L0_reg_ind,
        "C_0_rho": C_0_rho,
        "C_0_alpha": list(C_0_alpha),
    }

    if include_objval:
        indices.update({"objval_name": [OBJVAL_NAME], "objval": position[OBJVAL_NAME]})

    if include_L0_norm:
        indices.update({"L0_norm_name": [L0_NORM_NAME], "L0_norm": position[L0_NORM_NAME]})

    return indices


def cast_mip_start(values: Sequence[float], variable_types: Sequence[str]) -> list:
    """Cast solution values to the type of their MIP variable.

    Args:
        values: Solution values, aligned with ``variable_types``.
        variable_types: Variable type codes: ``"B"`` (binary), ``"I"`` (integer), or
            ``"C"`` (continuous).

    Returns:
        Values as ``int`` for binary/integer variables and ``float`` for continuous ones.
    """
    vals = list(values)
    for j, t in enumerate(variable_types):
        if t in ["B", "I"]:
            vals[j] = int(vals[j])
        elif t in ["C"]:
            vals[j] = float(vals[j])
    return vals


def convert_to_risk_slim_solution(
    rho: np.ndarray, indices: dict, loss: float | None = None, objval: float | None = None
) -> tuple[list[float], float]:
    """Convert coefficient vector ``rho`` into a full solution of the RiskSLIM MIP.

    Args:
        rho: Model coefficients.
        indices: Indices of the built RiskSLIM MIP (see ``build_mip_indices``).
        loss: Loss at ``rho``. Derived from ``objval`` when omitted.
        objval: Objective value at ``rho`` (loss plus L0 penalty). Derived from ``loss``
            when omitted.

    Returns:
        ``(values, objval)`` where ``values[k]`` is the value of MIP variable ``k`` for all
        ``indices["n_variables"]`` variables.

    Raises:
        ValueError: If the MIP has a loss or objval variable and neither ``loss`` nor
            ``objval`` is given.
    """
    n_variables = indices["n_variables"]
    solution_val = np.zeros(n_variables)

    # rho
    solution_val[indices["rho"]] = rho

    # alpha
    alpha = np.zeros(len(indices["alpha"]))
    alpha[np.flatnonzero(rho[indices["L0_reg_ind"]])] = 1.0
    solution_val[indices["alpha"]] = alpha
    L0_penalty = np.sum(indices["C_0_alpha"] * alpha)

    # add loss / objval
    need_objective_val = "objval" in indices
    need_L0_norm = "L0_norm" in indices

    if loss is None and objval is None:
        raise ValueError("convert_to_risk_slim_solution needs loss or objval")

    if loss is None:
        loss = objval - L0_penalty
    solution_val[indices["loss"]] = loss

    if need_objective_val:
        if objval is None:
            objval = loss + L0_penalty
        solution_val[indices["objval"]] = objval

    if need_L0_norm:
        solution_val[indices["L0_norm"]] = np.sum(alpha)

    return solution_val.tolist(), objval


# ---- abstract problem


class RiskSLIMMIP(ABC):
    """RiskSLIM surrogate MIP on a specific solver.

    Lifecycle: instantiate, ``build`` (sets ``self.indices``), optionally ``add_mip_starts``
    and ``register_callbacks``, ``set_parameters``, ``set_time_limit``, ``solve``, then read
    the solution. The warmstart cutting-plane loop (``run_standard_cpa``) builds the LP
    relaxation and alternates ``solve`` / ``add_cut`` / ``set_variable_bounds``.

    Variables are addressed by integer index, taken from ``self.indices`` (e.g.
    ``indices["rho"]``, ``indices["loss"]``, ``indices["objval"]``, ``indices["L0_norm"]``).

    Concrete subclasses hold their native solver object: the CPLEX backend exposes the
    ``cplex.Cplex`` as ``.cpx``. This class never touches a solver type.

    Attributes:
        indices: Indices of the built MIP (see ``build_mip_indices``); ``None`` before
            ``build``. Callers may add keys (the optimizer adds ``C_0_nnz`` and
            ``L0_reg_ind``); the dict is shared, not copied.
    """

    def __init__(self) -> None:
        """Create an empty problem; call ``build`` next."""
        self.indices: dict | None = None

    # ---- build and configure

    @abstractmethod
    def build(self, coef_set: CoefficientSet, settings: dict) -> dict:
        """Build the RiskSLIM surrogate MIP (no loss cuts yet).

        Must set ``self.indices`` and return that same dict. Variable and constraint names
        follow the module naming scheme; ``settings`` holds the formulation settings
        (``C_0``, bounds, ``relax_integer_variables``, ``drop_variables``,
        ``include_auxillary_variable_for_*``, ``set_cplex_cutoffs``) and receives their
        defaults in place.

        CPLEX: ``Cplex()``, ``variables.add``, ``linear_constraints.add/delete``,
            ``variables.delete``; LP via ``set_problem_type(LP)``; cutoffs via
            ``parameters.mip.tolerances.lower/uppercutoff``.
        SCIP: ``Model()``, ``addVar(name, vtype, lb, ub, obj)``, ``addCons(expr, name)``,
            ``setMinimize()``; skip dropped variables/constraints instead of deleting;
            relax via ``vtype="C"``; upper cutoff via ``setObjlimit``. Index order is the
            ``getVars()`` order of creation.

        Args:
            coef_set: Bounds, types, and penalties of the coefficients.
            settings: MIP formulation settings.

        Returns:
            ``self.indices``.
        """

    @abstractmethod
    def set_parameters(self, settings: dict, display_progress: bool = False) -> None:
        """Set solver parameters and silence solver output unless ``display_progress``.

        ``settings`` is the ``cplex_*`` block of the LCPA settings with the prefix removed
        (``randomseed``, ``n_cores``, ``mipemphasis``, ``mipgap``, ``absmipgap``,
        ``integrality_tolerance``, ``repairtries``, ``poolsize``, ``poolreplace``,
        ``optimality_tolerance``). MIP-only parameters apply only when the problem is a MIP.

        CPLEX: ``parameters.*.set``; output via ``parameters.mip.display``,
            ``parameters.simplex.display`` and ``set_*_stream(None)``.
        SCIP: ``setParam`` (``randomization/randomseedshift``, ``parallel/maxnthreads``,
            ``limits/gap``, ``limits/absgap``, ``numerics/feastol``,
            ``numerics/dualfeastol``, ``limits/maxsol``), ``setEmphasis``; no analog for
            ``repairtries`` / ``poolreplace``; output via ``hideOutput()``.

        Args:
            settings: Solver parameter settings.
            display_progress: Show solver progress output when True.
        """

    @abstractmethod
    def set_time_limit(self, seconds: float) -> None:
        """Limit the wall-clock time of the next ``solve``.

        CPLEX: ``parameters.timelimit.set``.
        SCIP: ``setParam("limits/time", seconds)``.

        Args:
            seconds: Time limit in seconds.
        """

    @abstractmethod
    def get_variable_bounds(self, idx: int) -> tuple[float, float]:
        """Return ``(lb, ub)`` of variable ``idx``.

        Used for the ``loss``, ``objval``, and ``L0_norm`` variables.

        CPLEX: ``variables.get_lower_bounds`` / ``get_upper_bounds``.
        SCIP: ``var.getLbOriginal()`` / ``var.getUbOriginal()``.

        Args:
            idx: Variable index.

        Returns:
            Lower and upper bound.
        """

    @abstractmethod
    def set_variable_bounds(self, idx: int, lb: float, ub: float) -> None:
        """Set the bounds of variable ``idx`` (lower bound first, then upper bound).

        Called between solves of the warmstart LP.

        CPLEX: ``variables.set_lower_bounds`` then ``set_upper_bounds``.
        SCIP: ``freeTransform()`` then ``chgVarLb`` / ``chgVarUb``.

        Args:
            idx: Variable index.
            lb: New lower bound.
            ub: New upper bound.
        """

    @abstractmethod
    def add_cut(self, idx: Sequence[int], coefs: Sequence[float], rhs: float) -> None:
        """Add the linear constraint ``sum_k coefs[k] * x[idx[k]] >= rhs`` to the model.

        Used by the warmstart cutting-plane loop to add loss cuts between solves. Lazy cuts
        during branch-and-bound go through the callbacks, not this method.

        CPLEX: ``linear_constraints.add(lin_expr=[SparsePair(idx, coefs)], senses=["G"],
            rhs=[rhs])``.
        SCIP: ``freeTransform()`` then ``addCons(quicksum(c * x) >= rhs)``; keep the
            returned constraint for ``get_cuts``.

        Args:
            idx: Variable indices.
            coefs: Coefficients, aligned with ``idx``.
            rhs: Right-hand side.
        """

    @abstractmethod
    def get_cuts(self) -> dict:
        """Return every constraint added by ``add_cut``, in order.

        The result is passed as ``initial_cuts`` to ``register_callbacks`` of a MIP on the
        same solver, so each row may be in the solver's native row format.

        CPLEX: rows ``indices["n_constraints"]`` to ``linear_constraints.get_num()`` via
            ``get_rows`` (``SparsePair``) and ``get_rhs``.
        SCIP: the stored constraints via ``getValsLinear`` and ``getLhs``.

        Returns:
            ``{"coefs": [row, ...], "lhs": [rhs, ...]}``.
        """

    @abstractmethod
    def add_mip_starts(self, pool: SolutionPool, max_mip_starts: float = float("inf")) -> None:
        """Add the distinct solutions of ``pool`` as MIP starts, best objective first.

        Skips solutions whose objective value exceeds the upper cutoff. Each start is the
        output of ``convert_to_risk_slim_solution`` cast with ``cast_mip_start``.

        CPLEX: ``MIP_starts.add(SparsePair, effort_level.repair, name)``; cutoff from
            ``parameters.mip.tolerances.uppercutoff``.
        SCIP: ``createSol()``, ``setSolVal`` per variable, ``addSol``; cutoff from
            ``getObjlimit()``. Must run before ``solve``.

        Args:
            pool: Pool of RiskSLIM coefficient vectors and objective values.
            max_mip_starts: Maximum number of starts to add.
        """

    @abstractmethod
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
    ) -> tuple[object, object | None]:
        """Attach the lattice-CPA loss-cut callback and, if enabled, the heuristic callback.

        The loss-cut callback runs at every integer-feasible candidate: it adds the loss cut
        at ``rho`` (from ``compute_loss_cut``), adds ``initial_cuts`` on its first call, adds
        cuts at solutions in ``cut_queue``, sends candidates to ``polish_queue``, tightens
        bounds when ``settings["chained_updates_flag"]``, and records progress in ``stats``.
        The heuristic callback is attached iff ``settings["round_flag"]`` or
        ``settings["polish_flag"]``; it rounds (``rounding_handle``) and polishes
        (``polishing_handle``) and proposes improving solutions built with
        ``convert_to_risk_slim_solution``. ``settings`` is shared and may be mutated during
        the search. Uses ``self.indices``, which must already contain ``C_0_nnz`` and
        ``L0_reg_ind``.

        CPLEX: ``register_callback(LossCallback)`` (a ``LazyConstraintCallback``) and
            ``register_callback(PolishAndRoundCallback)`` (a ``HeuristicCallback``), each
            followed by ``initialize(...)``.
        SCIP: a ``Conshdlr`` with ``needscons=False`` whose ``conscheck`` rejects candidates
            whose loss variable underestimates the loss and whose ``consenfolp`` /
            ``consenfops`` add the cut (``addCons`` or a row via ``addCut``, ``removable``
            for purgeable cuts), registered with ``includeConshdlr``; a ``Heur`` whose
            ``heurexec`` builds a solution with ``createSol`` / ``setSolVal`` and submits it
            with ``trySol``, registered with ``includeHeur``. Progress stats come from
            ``getDualbound``, ``getPrimalbound``, ``getGap``, ``getNNodes``,
            ``getNNodesLeft``.

        Args:
            stats: Shared search statistics and bounds.
            settings: LCPA settings.
            compute_loss_cut: ``rho -> (loss_value, loss_slope)``.
            get_alpha: ``rho -> alpha``.
            get_L0_penalty_from_alpha: ``alpha -> L0 penalty``.
            cut_queue: Solutions at which to add loss cuts.
            polish_queue: Solutions to polish.
            initial_cuts: Cuts from the warmstart LP (``get_cuts`` output), or ``None``.
            get_objval: ``rho -> objective value``; heuristic only.
            get_L0_norm: ``rho -> L0 norm``; heuristic only.
            is_feasible: ``rho -> bool``; heuristic only.
            polishing_handle: ``rho -> (polished_rho, loss, objval)``; heuristic only.
            rounding_handle: ``(rho, cutoff) -> (rounded_rho, objval, early_stop)``;
                heuristic only.
            verbose: Log callback activity.

        Returns:
            ``(cut_callback, heuristic_callback)``; ``heuristic_callback`` is ``None`` when
            not attached.
        """

    # ---- solve and read the solution

    @abstractmethod
    def solve(self) -> None:
        """Solve the model in place.

        CPLEX: ``solve()``.
        SCIP: ``optimize()``.
        """

    @abstractmethod
    def has_solution(self) -> bool:
        """Return True if the last solve found a feasible solution.

        Guards ``get_values`` / ``objective_value`` / ``best_bound`` / ``relative_gap``.

        CPLEX: ``solution.is_primal_feasible()``.
        SCIP: ``getNSols() > 0``.
        """

    @abstractmethod
    def is_optimal(self) -> bool:
        """Return True if the last solve ended optimal (within tolerance).

        CPLEX: status name in ``riskslim.opt.cpx.solver.OPTIMAL_STATUS_NAMES``.
        SCIP: ``getStatus() == "optimal"``.
        """

    @abstractmethod
    def status_name(self) -> str:
        """Return the solver's short status name of the last solve.

        CPLEX: ``solution.status[solution.get_status()]`` (e.g. ``"MIP_optimal"``).
        SCIP: ``getStatus()`` (e.g. ``"optimal"``, ``"timelimit"``).
        """

    @abstractmethod
    def status_string(self) -> str:
        """Return the solver's human-readable status of the last solve.

        CPLEX: ``solution.get_status_string()`` (e.g. ``"integer optimal solution"``).
        SCIP: ``str(getStatus())``.
        """

    @abstractmethod
    def get_values(self, idx: Sequence[int]) -> np.ndarray:
        """Return the values of variables ``idx`` in the best solution.

        CPLEX: ``solution.get_values(idx)``.
        SCIP: ``getSolVal(getBestSol(), var)`` for each variable.

        Args:
            idx: Variable indices.

        Returns:
            1d array of values, aligned with ``idx``.
        """

    @abstractmethod
    def objective_value(self) -> float:
        """Return the objective value of the best solution (LP: the LP optimum).

        CPLEX: ``solution.get_objective_value()``.
        SCIP: ``getObjVal()``.
        """

    @abstractmethod
    def best_bound(self) -> float:
        """Return the best lower bound on the MIP objective.

        CPLEX: ``solution.MIP.get_best_objective()``.
        SCIP: ``getDualbound()``.
        """

    @abstractmethod
    def relative_gap(self) -> float:
        """Return the relative MIP gap between the best solution and the best bound.

        CPLEX: ``solution.MIP.get_mip_relative_gap()``
            (``|bound - incumbent| / (1e-10 + |incumbent|)``).
        SCIP: ``getGap()`` (divides by ``min(|primal|, |dual|)``, so values differ from
            CPLEX's).
        """

    @abstractmethod
    def simplex_iteration_count(self) -> int:
        """Return the number of simplex iterations of the last solve.

        CPLEX: ``solution.progress.get_num_iterations()``.
        SCIP: ``getNLPIterations()``.
        """


def load(solver: str = "cplex") -> type[RiskSLIMMIP]:
    """Return the ``RiskSLIMMIP`` subclass for ``solver``, importing its backend lazily.

    Args:
        solver: Solver name. Only ``"cplex"`` is available.

    Returns:
        The concrete ``RiskSLIMMIP`` class.

    Raises:
        ImportError: If the solver's Python package is not installed.
        ValueError: If ``solver`` is unknown.
    """
    if solver == "cplex":
        check_cplex()
        from riskslim.opt.cpx.solver import CplexRiskSLIMMIP

        return CplexRiskSLIMMIP
    raise ValueError(f"unknown solver: {solver}")
