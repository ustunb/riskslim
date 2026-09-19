"""Bounded checks of the CPLEX behaviors used by RiskSLIM.

Test strategy:
    * model domain: continuous LP and binary MIP; the domain changes the
      optimal vector and whether integrality and a MIP bound apply.
    * lazy separation: callback disabled versus enabled; a MIP start is n/a
      for the LP and is supplied only for the binary callback case.
    * MIP start: absent versus a base-feasible, lazy-invalid incumbent; both
      must produce the same lazy-constrained optimum.

These tests intentionally exercise the installed CPLEX API directly. They do
not import RiskSLIM or assert any production bookkeeping.
"""

import cplex
import pytest


class LazyUpperBoundCallback(cplex.callbacks.LazyConstraintCallback):
    """Separate the candidate x <= 0 constraint when x is selected."""

    def __call__(self):
        if self.get_values("x") > 0.5:
            self.add(
                constraint=cplex.SparsePair(ind=["x"], val=[1.0]),
                sense="L",
                rhs=0.0,
            )


def build_bounded_model(variable_type):
    """Build the small LP/MIP shared by the direct solver checks."""
    model = cplex.Cplex()
    model.set_results_stream(None)
    model.set_log_stream(None)
    variable_settings = {
        "names": ["x", "y"],
        "obj": [3.0, 2.0],
        "lb": [0.0, 0.0],
        "ub": [1.0, 1.0],
    }
    if variable_type == "B":
        variable_settings["types"] = "BB"
    model.variables.add(**variable_settings)
    model.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=["x", "y"], val=[2.0, 2.0])],
        senses=["L"],
        rhs=[3.0],
    )
    model.objective.set_sense(model.objective.sense.maximize)
    return model


@pytest.mark.parametrize(
    "variable_type, expected_values, expected_objective",
    [
        pytest.param("C", [1.0, 0.5], 4.0, id="continuous-lp"),
        pytest.param("B", [1.0, 0.0], 3.0, id="binary-mip"),
    ],
)
def test_cplex_solves_lp_and_binary_mip_with_expected_optima(
    variable_type, expected_values, expected_objective
):
    """Continuous and binary domains produce their known optimal solutions."""
    model = build_bounded_model(variable_type)
    try:
        model.solve()
        expected_status = (
            model.solution.status.optimal
            if variable_type == "C"
            else model.solution.status.MIP_optimal
        )
        expected_problem_type = (
            model.problem_type.LP if variable_type == "C" else model.problem_type.MILP
        )
        assert model.get_problem_type() == expected_problem_type
        assert model.solution.get_status() == expected_status
        values = model.solution.get_values()
        assert values == pytest.approx(expected_values, rel=0.0, abs=1e-9)
        assert all(0.0 <= value <= 1.0 for value in values)
        assert model.solution.get_objective_value() == pytest.approx(
            expected_objective, rel=0.0, abs=1e-9
        )
        assert sum(2.0 * value for value in values) <= 3.0 + 1e-9
        assert sum(
            coefficient * value for coefficient, value in zip([3.0, 2.0], values)
        ) == pytest.approx(model.solution.get_objective_value(), rel=0.0, abs=1e-9)
        if variable_type == "B":
            assert all(value == pytest.approx(round(value), rel=0.0, abs=1e-9) for value in values)
            assert model.solution.MIP.get_best_objective() == pytest.approx(3.0, rel=0.0, abs=1e-9)
    finally:
        model.end()


@pytest.mark.parametrize("use_mip_start", [False, True], ids=["no-start", "invalid-lazy-start"])
def test_cplex_lazy_callback_enforces_constraint_with_or_without_mip_start(use_mip_start):
    """A lazy cut changes the known base-MIP optimum and survives a start."""
    model = build_bounded_model("B")
    try:
        model.register_callback(LazyUpperBoundCallback)
        if use_mip_start:
            model.MIP_starts.add([["x", "y"], [1.0, 0.0]], model.MIP_starts.effort_level.auto)
        model.solve()

        assert model.solution.get_status() == model.solution.status.MIP_optimal
        assert model.solution.get_values() == pytest.approx([0.0, 1.0], rel=0.0, abs=1e-9)
        assert model.solution.get_objective_value() == pytest.approx(2.0, rel=0.0, abs=1e-9)
        assert model.solution.get_values("x") <= 1e-9
        assert model.solution.MIP.get_best_objective() == pytest.approx(2.0, rel=0.0, abs=1e-9)
    finally:
        model.end()
