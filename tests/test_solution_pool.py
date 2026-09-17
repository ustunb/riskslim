"""Test solution pools."""

import pytest
import numpy as np
from riskslim.solution_pool import SolutionPool, FastSolutionPool


@pytest.mark.parametrize('init',
    [
        'SolutionPool',
        'int',
        'dict0',
        'dict1',
        pytest.param('dict2', marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param(None, marks=pytest.mark.xfail(raises=ValueError))
    ]
)
def test_solution_pool_init(init):

    if init == 'SolutionPool':
        init_pool = SolutionPool(10)
        pool = SolutionPool(init_pool)
    elif init == 'int':
        pool = SolutionPool(10)
    elif init == 'dict0':
        solutions = np.zeros(10)
        objval = np.random.rand(1)
        pool = SolutionPool({'solutions': solutions, 'objvals': objval})
    elif init == 'dict1':
        solutions = np.zeros((2, 10)).T
        objval = np.random.rand(2)
        pool = SolutionPool({'solutions': solutions, 'objvals': objval})
    elif init == 'dict2':
        # Incorrect shape
        solutions = np.zeros((1, 2, 10))
        objval = np.random.rand(2)
        pool = SolutionPool({'solutions': solutions, 'objvals': objval})
    else:
        # Expects error
        pool = SolutionPool(None)

    assert pool._P == 10
    assert pool._solutions.shape[-1] == 10
    assert len(pool._objvals) in [0, 1, 2]

    assert len(pool) == len(pool._objvals)


def test_solution_pool_solution_string():

    solutions = np.zeros(10) + 1e-6
    objval = np.random.rand(1)
    pool = SolutionPool({'solutions': solutions, 'objvals': objval})

    solution_string = SolutionPool.solution_string(pool._solutions[0])

    for i in solution_string:
        assert i in [' ', '0', '.']


def test_solution_pool_solution_table():

    solutions = np.zeros(10)
    objval = np.random.rand(1)
    pool = SolutionPool({'solutions': solutions, 'objvals': objval})

    table = pool.table()
    assert isinstance(table, str)
    assert 'solution' in table
    assert 'objval' in table

    assert table == pool.__repr__() == pool.__str__()


def test_solution_pool_copy():

    solutions = np.zeros(10)
    objval = np.random.rand(1)
    pool = SolutionPool({'solutions': solutions, 'objvals': objval})
    pool_cp = pool.copy()

    assert isinstance(pool_cp, type(pool))


def test_solution_pool_properties():

    solutions = np.zeros(10)
    objval = np.random.rand(1)
    pool = SolutionPool({'solutions': solutions, 'objvals': objval})

    # Getters
    assert pool.P == 10
    assert pool.objvals[0] >= 0 and pool.objvals[0] <= 1
    assert pool.solutions.shape[-1] == len(solutions)
    assert np.all(pool.solutions[0] == 0)

    # Setters
    pool.objvals = -1
    assert pool.objvals == -1
    pool.objvals = [-1]
    assert pool.objvals[0] == -1
    pool.objvals = []

    pool.solutions = np.ones(10)
    assert np.all(pool.solutions[0] == 1)

    pool.solutions = np.ones((1, 10))
    assert np.all(pool.solutions[0] == 1)

    pool.solutions = np.ones((1, 10)).T
    assert np.all(pool.solutions[0] == 1)

    with pytest.raises(ValueError):
        pool.solutions = np.ones((1, 1, 10))


def test_solution_pool_append():

    pool = SolutionPool(10)

    # Append
    pool.append([])
    assert len(pool) == 0

    solutions = np.zeros(10)
    objval = np.random.rand(1)

    pool.append(SolutionPool({'solutions': solutions, 'objvals': objval}))

    assert len(pool) == 1


@pytest.mark.parametrize('objval_solution', [
        ([0.], np.zeros(10)),
        ([0.], np.zeros((1, 10))),
        ([0.], np.zeros((10, 1))),
        pytest.param(([0.], np.zeros((1, 2, 3))), marks=pytest.mark.xfail(raises=ValueError)),
        ([0.], np.zeros((1, 10)).tolist()),
        pytest.param(([0.], None), marks=pytest.mark.xfail(raises=TypeError)),
        (0., np.zeros(10)),
        ([], np.zeros(10)),
    ]
)
def test_solution_pool_add(objval_solution):

    objvals, solutions = objval_solution

    pool = SolutionPool(10)
    pool.add(objvals, solutions)
    assert len(pool) == 1 or len(pool) == 0


def test_solution_pool_filter():
    solutions = np.zeros((2, 10))
    objval = np.random.rand(2)
    pool = SolutionPool({'solutions': solutions, 'objvals': objval})

    assert len(pool) == 2

    pool.filter([True, False])
    assert len(pool) == 1

    pool = SolutionPool({'solutions': solutions, 'objvals': objval})
    pool.filter([False, False])
    assert len(pool) == 0


def test_solution_pool_distinct():

    solutions = np.zeros((2, 10))
    objval = np.random.rand(2)
    pool = SolutionPool({'solutions': solutions, 'objvals': objval})

    assert len(pool) == 2

    pool.distinct()

    assert len(pool) == 1


def test_solution_pool_sort():

    solutions = np.zeros((2, 10))
    objval = np.array([1., 0.])
    pool = SolutionPool({'solutions': solutions, 'objvals': objval})

    pool.sort()
    assert np.all(pool.objvals == objval[::-1])


@pytest.mark.parametrize('target',
    [
        'all', 'objvals', 'solutions',
        pytest.param(None, marks=pytest.mark.xfail(raises=ValueError)),
    ]
)
def test_solution_pool_map(target):

    solutions = np.zeros((2, 10))
    objval = np.array([1., 0.])
    pool = SolutionPool({'solutions': solutions, 'objvals': objval})

    if target == 'all':
        map_func = lambda i, j: 0
    else:
        map_func = lambda i: 0

    vals = pool.map(map_func, target)
    vals = np.array(vals)
    assert all(vals == 0)


def test_solution_pool_remove_nonintegral():
    solutions = np.zeros((2, 10)) + .01
    objval = [1, 0.]
    pool = SolutionPool({'solutions': solutions, 'objvals': objval})
    pool.remove_nonintegral()
    assert len(pool) == 0


def test_solution_pool_compute_objvals():
    solutions = np.zeros((2, 10))
    objval = [np.nan, np.nan]
    pool = SolutionPool({'solutions': solutions, 'objvals': objval})
    get_objval = lambda i: 0
    pool.compute_objvals(get_objval)
    assert all(pool.objvals == 0)


def test_solution_pool_remove_suboptimal():
    solutions = np.zeros((2, 10))
    objval = [1, 1]
    pool = SolutionPool({'solutions': solutions, 'objvals': objval})
    pool.remove_suboptimal(0)
    assert len(pool) == 0


def test_fast_solution_pool():

    solution_pool = FastSolutionPool(1)
    assert solution_pool.P == 1
    assert len(solution_pool.objvals) == 0
    assert len(solution_pool.solutions) == 0

    assert solution_pool.table() == solution_pool.__repr__() == solution_pool.__str__()


def test_fast_solution_pool_add():
    solution_pool = FastSolutionPool(2)
    solution_pool.add(np.array([0, 1]), np.array([[0, 0], [1, 1]]))
    assert len(solution_pool.objvals) == 2

    solution_pool = FastSolutionPool(2)
    solution_pool.add(0, np.array([0, 0]))
    assert len(solution_pool.objvals) == 1


def test_fast_solution_pool_get_best():

    solution_pool = FastSolutionPool(2)
    solution_pool.add(np.array([0, 1]), np.array([[0, 0], [1, 1]]))
    objval, solutions = solution_pool.get_best_objval_and_solution()

    assert objval == 0
    assert all(solutions == 0)

    solution_pool = FastSolutionPool(2)
    objval, solutions = solution_pool.get_best_objval_and_solution()
    assert len(objval) == 0
    assert len(solutions) == 0


def test_fast_solution_pool_solution_string():

    solution_pool = FastSolutionPool(2)
    solution_pool.add(np.array([0]), np.array([0, 0]))
    solution_string = FastSolutionPool.solution_string(solution_pool.solutions[0])
    for s in solution_string:
        assert s in [' ', '0']

    solution_pool = FastSolutionPool(2)
    solution_pool.add(np.array([0]), np.array([0.1, 0.1]))
    solution_string = FastSolutionPool.solution_string(solution_pool.solutions[0])
    for s in solution_string:
        assert s in [' ', '0', '1', '.']
