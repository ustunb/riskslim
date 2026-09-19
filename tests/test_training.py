# Test strategy: exact synthetic training oracles
#
# Dimensions:
#   features: binary, continuous, mixed — exercise distinct score distributions
#   truth: integer, fractional, shifted fractional — exercise coefficient/intercept recovery
#   model bounds: risk-score [-5, 5], checklist [-1, 1] — distinct feasible model sets
#   warm start: off, on — distinct solver initialization paths; lazy loss cuts stay enabled
#   size path: 4/None, 2, 1, 0 — fresh fits as the model-size limit tightens
#   penalty path: 1e-6, .01, .1, 1 — fresh fits as sparsity pressure increases
#   redundant features: none, duplicates, noisy duplicates — joint penalty-plus-redundancy behavior
# n/a: noisy duplicates x risk-score — exact enumeration would require 11^8 states.

import numpy as np
import pytest

from riskslim import RiskSLIMClassifier


BASELINE_CASE_IDS = tuple(
    f"{feature_distribution}__{truth_name}"
    for feature_distribution in ("binary", "continuous", "mixed")
    for truth_name in ("integer", "fractional", "shifted_fractional")
)
MODEL_COEFFICIENT_BOUNDS = {"risk-score": 5, "checklist": 1}
REDUNDANCY_MODEL_CASES = (
    ("none", "risk-score"),
    ("none", "checklist"),
    ("duplicates", "risk-score"),
    ("duplicates", "checklist"),
    ("noisy_duplicates", "checklist"),
)
OPTIMAL_STATUSES = {"integer optimal solution", "integer optimal, tolerance"}
OBJECTIVE_TOLERANCE = 1e-6


def get_oracle(case, model_type, c0_value, max_size):
    task = case["tasks"][model_type]
    regeneration_command = (
        f"uv run python tests/generate_training_test_cases.py --regenerate --case {case['case_id']}"
    )
    query = task["queries"].get((c0_value, max_size))
    assert query is not None, (
        f"Precondition: {case['case_id']} {model_type} query is missing. Run "
        f"`{regeneration_command}`."
    )
    assert case["data_digest"] == case["data_identity"]["data_digest"], (
        f"Precondition: {case['case_id']} has a stale data identity. Run `{regeneration_command}`."
    )
    assert task["loss_identity"]["data_digest"] == case["data_digest"], (
        f"Precondition: {case['case_id']} {model_type} has a stale loss identity. Run "
        f"`{regeneration_command}`."
    )
    assert query["query_identity"] == {
        "loss_identity_digest": task["loss_identity_digest"],
        "c0": c0_value,
        "max_size": max_size,
    }, f"Precondition: {case['case_id']} {model_type} query is stale. Run `{regeneration_command}`."
    assert query["proven_complete"], (
        f"Precondition: {case['case_id']} {model_type} query is not exact. Run "
        f"`{regeneration_command}`."
    )
    oracle_rho = query["representative_rho"]
    oracle_support = int(np.count_nonzero(oracle_rho[1:]))
    oracle_signed_scores = (1.0 - 2.0 * case["y"]) * (oracle_rho[0] + case["X"] @ oracle_rho[1:])
    oracle_loss = float(np.mean(np.logaddexp(0.0, oracle_signed_scores)))
    oracle_objective = oracle_loss + c0_value * oracle_support
    assert np.isfinite(oracle_rho).all()
    np.testing.assert_allclose(oracle_rho, np.rint(oracle_rho), rtol=0.0, atol=0.0)
    assert np.all(oracle_rho >= task["coefficient_set_lower_bounds"])
    assert np.all(oracle_rho <= task["coefficient_set_upper_bounds"])
    assert oracle_support <= max_size
    np.testing.assert_allclose(query["pure_logistic_loss"], oracle_loss, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(query["objective"], oracle_objective, rtol=0.0, atol=1e-12)
    assert query["representative_tie_count"] >= 1, (
        f"Precondition: {case['case_id']} {model_type} query has no optimum. Run "
        f"`{regeneration_command}`."
    )
    return task, query


def assert_loss_does_not_decrease(earlier_result, later_result):
    loss_change = later_result["loss"] - earlier_result["loss"]
    assert loss_change >= -OBJECTIVE_TOLERANCE

    earlier_oracle = earlier_result["oracle"]
    later_oracle = later_result["oracle"]
    oracle_loss_change = later_oracle["pure_logistic_loss"] - earlier_oracle["pure_logistic_loss"]
    both_oracles_are_unique = (
        earlier_oracle["representative_tie_count"] == 1
        and later_oracle["representative_tie_count"] == 1
    )
    if both_oracles_are_unique and oracle_loss_change > OBJECTIVE_TOLERANCE:
        assert loss_change > 0.0


def fit_and_assert_global_optimum(
    case, model_type, warm_start, c0_value, max_size, oracle_max_size
):
    task, oracle = get_oracle(case, model_type, c0_value, oracle_max_size)
    classifier = RiskSLIMClassifier(
        initialization_flag=warm_start,
        max_coef=MODEL_COEFFICIENT_BOUNDS[model_type],
        max_size=max_size,
        c0_value=c0_value,
        variable_names=list(case["variable_names"]),
        outcome_name=case["outcome_name"],
        verbose=False,
        cplex_randomseed=0,
    )
    classifier.fit(case["X"], case["y"])

    optimizer = classifier.optimizer
    solution = optimizer.solution
    solution_info = optimizer.solution_info
    rho = np.r_[classifier.intercept_, classifier.coef_]
    support = int(np.count_nonzero(rho[1:]))
    signed_scores = (1.0 - 2.0 * case["y"]) * (rho[0] + case["X"] @ rho[1:])
    pure_logistic_loss = float(np.mean(np.logaddexp(0.0, signed_scores)))
    recomputed_objective = pure_logistic_loss + c0_value * support
    raw_objective = solution.get_objective_value()
    best_objective = solution.MIP.get_best_objective()

    np.testing.assert_allclose(
        optimizer.coef_set.lb,
        task["coefficient_set_lower_bounds"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        optimizer.coef_set.ub,
        task["coefficient_set_upper_bounds"],
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isfinite(rho).all()
    np.testing.assert_allclose(rho, np.rint(rho), rtol=0.0, atol=1e-8)
    assert np.all(rho >= task["coefficient_set_lower_bounds"])
    assert np.all(rho <= task["coefficient_set_upper_bounds"])
    assert support <= oracle_max_size
    np.testing.assert_allclose(
        recomputed_objective, oracle["objective"], rtol=0.0, atol=OBJECTIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        raw_objective, recomputed_objective, rtol=0.0, atol=OBJECTIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        solution_info["loss_value"], pure_logistic_loss, rtol=0.0, atol=OBJECTIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        solution_info["objective_value"],
        recomputed_objective,
        rtol=0.0,
        atol=OBJECTIVE_TOLERANCE,
    )
    assert solution.get_status_string() in OPTIMAL_STATUSES
    assert optimizer.stats.cplex_status in OPTIMAL_STATUSES
    assert np.isfinite(best_objective)
    np.testing.assert_allclose(
        best_objective,
        recomputed_objective,
        rtol=OBJECTIVE_TOLERANCE,
        atol=OBJECTIVE_TOLERANCE,
    )
    return {
        "loss": pure_logistic_loss,
        "objective": recomputed_objective,
        "support": support,
        "rho": rho,
        "oracle": oracle,
    }


@pytest.mark.parametrize("warm_start", [False, True], ids=["warm-off", "warm-on"])
@pytest.mark.parametrize("model_type", ["risk-score", "checklist"])
@pytest.mark.parametrize("case_id", BASELINE_CASE_IDS)
def test_solver_returns_global_optimum_with_model_size_limit(
    training_test_cases, case_id, model_type, warm_start
):
    case = training_test_cases["synthetic_oracles"][case_id]
    maximum_result = fit_and_assert_global_optimum(case, model_type, warm_start, 1e-6, 4, 4)
    path = [maximum_result]
    for max_size in (2, 1, 0):
        path.append(
            fit_and_assert_global_optimum(case, model_type, warm_start, 1e-6, max_size, max_size)
        )

    if case_id == "binary__integer" and model_type == "risk-score" and not warm_start:
        none_result = fit_and_assert_global_optimum(case, model_type, warm_start, 1e-6, None, 4)
        np.testing.assert_allclose(
            none_result["objective"], maximum_result["objective"], rtol=0.0, atol=1e-6
        )
    for less_restricted, more_restricted in zip(path, path[1:], strict=False):
        assert_loss_does_not_decrease(less_restricted, more_restricted)


@pytest.mark.parametrize("warm_start", [False, True], ids=["warm-off", "warm-on"])
@pytest.mark.parametrize("model_type", ["risk-score", "checklist"])
@pytest.mark.parametrize("case_id", BASELINE_CASE_IDS)
def test_solver_returns_global_optimum_with_sparsity_penalty(
    training_test_cases, case_id, model_type, warm_start
):
    case = training_test_cases["synthetic_oracles"][case_id]
    path = [
        fit_and_assert_global_optimum(case, model_type, warm_start, c0_value, 4, 4)
        for c0_value in (1e-6, 0.01, 0.1, 1.0)
    ]

    for lower_penalty, higher_penalty in zip(path, path[1:], strict=False):
        assert higher_penalty["support"] <= lower_penalty["support"]
        assert_loss_does_not_decrease(lower_penalty, higher_penalty)


@pytest.mark.parametrize("warm_start", [False, True], ids=["warm-off", "warm-on"])
@pytest.mark.parametrize("model_type", ["risk-score", "checklist"])
@pytest.mark.parametrize("case_id", BASELINE_CASE_IDS)
def test_solver_returns_global_optimum_with_coefficient_bounds(
    training_test_cases, case_id, model_type, warm_start
):
    case = training_test_cases["synthetic_oracles"][case_id]
    fit_and_assert_global_optimum(case, model_type, warm_start, 1e-6, 4, 4)


@pytest.mark.parametrize("warm_start", [False, True], ids=["warm-off", "warm-on"])
@pytest.mark.parametrize(
    "redundant_features,model_type",
    REDUNDANCY_MODEL_CASES,
    ids=[
        "none-risk-score",
        "none-checklist",
        "duplicates-risk-score",
        "duplicates-checklist",
        "noisy-duplicates-checklist",
    ],
)
@pytest.mark.parametrize("case_id", BASELINE_CASE_IDS)
def test_solver_drops_redundant_features_at_global_optimum(
    training_test_cases, case_id, redundant_features, model_type, warm_start
):
    if redundant_features == "none":
        case = training_test_cases["synthetic_oracles"][case_id]
        fit_and_assert_global_optimum(case, model_type, warm_start, 0.01, 4, 4)
        return

    redundant_case_id = f"{case_id}__{redundant_features}"
    case = training_test_cases["synthetic_oracles"][redundant_case_id]
    task = case["tasks"][model_type]
    c0_value, max_size = next(iter(task["queries"]))
    oracle = task["queries"][(c0_value, max_size)]
    assert np.count_nonzero(oracle["representative_rho"][1:]) > 0
    result = fit_and_assert_global_optimum(
        case, model_type, warm_start, c0_value, max_size, max_size
    )
    feature_coefficients = result["rho"][1:]
    assert result["support"] > 0
    assert not np.any((feature_coefficients[:4] != 0.0) & (feature_coefficients[4:] != 0.0))
