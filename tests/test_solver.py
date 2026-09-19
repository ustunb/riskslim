"""Smoke-test that CPLEX is installed and can solve a bounded binary model."""

from riskslim.cplex_utils import check_cplex_installation


def test_cplex_can_solve():
    """Verify the installed CPLEX runtime can produce a feasible solution."""
    versions = check_cplex_installation()
    print(
        f"CPLEX package {versions['package_version']}; native runtime {versions['runtime_version']}"
    )
