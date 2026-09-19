"""Smoke-test that CPLEX is installed and can solve a bounded binary model."""

import cplex


def test_cplex_can_solve():
    """Verify the installed CPLEX runtime can produce a feasible solution."""
    with cplex.Cplex() as model:
        model.set_results_stream(None)
        model.set_log_stream(None)
        model.variables.add(names=["x"], types=["B"], lb=[0.0], ub=[1.0])
        model.solve()

        assert model.solution.is_primal_feasible()
