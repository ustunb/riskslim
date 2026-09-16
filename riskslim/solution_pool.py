import numpy as np
import prettytable as pt

class SolutionPool(object):
    """
    Helper class used to store solutions to the risk slim optimization problem
    """

    def __init__(self,  obj):

        if isinstance(obj, SolutionPool):

            self._P = obj.P
            self._objvals = obj.objvals
            self._solutions = obj.solutions

        elif isinstance(obj, int):

            assert obj >= 1
            self._P = int(obj)
            self._objvals = np.empty(0)
            self._solutions = np.empty(shape = (0, self._P))

        elif isinstance(obj, dict):

            assert len(obj) == 2
            objvals = np.copy(obj['objvals']).flatten().astype(dtype = np.float_)
            solutions = np.copy(obj['solutions'])
            n = objvals.size
            if solutions.ndim == 2:
                assert n in solutions.shape
                if solutions.shape[1] == n and solutions.shape[0] != n:
                    solutions = np.transpose(solutions)
            elif solutions.ndim == 1:
                assert n == 1
                solutions = np.reshape(solutions, (1, solutions.size))
            else:
                raise ValueError('solutions has more than 2 dimensions')

            self._P = solutions.shape[1]
            self._objvals = objvals
            self._solutions = solutions

        else:
            raise ValueError('cannot initialize SolutionPool using %s object' % type(obj))


    def __len__(self):
        return len(self._objvals)


    @staticmethod
    def solution_string(solution, float_fmt = '%1.3f'):
        solution_string = ''
        for j in range(len(solution)):
            if SolutionPool.is_integral(solution[j]):
                solution_string += ' ' + str(int(solution[j]))
            else:
                solution_string += ((' ' + float_fmt) % solution[j])
        return solution_string


    def table(self):
        x = pt.PrettyTable(align = 'r', float_format = '1.3', hrules = pt.ALL)
        x.add_column("objval", self._objvals.tolist())
        x.add_column("solution", list(map(self.solution_string, self._solutions)))
        return str(x)


    def __repr__(self):
        return self.table()


    def __str__(self):
        return self.table()


    def copy(self):
        return SolutionPool(self)


    @property
    def P(self):
        return int(self._P)


    @property
    def objvals(self):
        return self._objvals


    @property
    def solutions(self):
        return self._solutions


    @objvals.setter
    def objvals(self, objvals):
        if hasattr(objvals, "__len__"):
            if len(objvals) > 0:
                self._objvals = np.copy(list(objvals)).flatten().astype(dtype = np.float_)
            elif len(objvals) == 0:
                self._objvals = np.empty(0)
        else:
            self._objvals = float(objvals)


    @solutions.setter
    def solutions(self, solutions):
        if solutions.ndim == 2:
            assert self._P in solutions.shape
            if solutions.shape[0] == self._P and solutions.shape[1] != self._P:
                solutions = np.transpose(solutions)
        elif solutions.ndim == 1:
            solutions = np.reshape(solutions, (1, solutions.size))
        else:
            raise ValueError('incorrect solution dimensions')

        self._solutions = np.copy(solutions)


    def append(self, pool):
        if len(pool) == 0:
            return self
        else:
            return self.add(pool.objvals, pool.solutions)


    def add(self, objvals, solutions):

        if isinstance(objvals, np.ndarray) or isinstance(objvals, list):
            n = len(objvals)
            if n == 0:
                return self
            if isinstance(solutions, np.ndarray):
                if solutions.ndim == 2:
                    assert n in solutions.shape
                    assert self._P in solutions.shape
                    if solutions.shape[0] == self._P and solutions.shape[1] != self._P:
                        solutions = np.transpose(solutions)
                elif solutions.ndim == 1:
                    assert n == 1
                    solutions = np.reshape(solutions, (1, solutions.size))
                else:
                    raise ValueError('incorrect solution dimensions')
            elif isinstance(solutions, list):
                solutions = np.array(solutions)
                assert solutions.shape[0] == n
                assert solutions.shape[1] == self._P
            else:
                raise TypeError('incorrect solution type')
        else:
            objvals = float(objvals) #also assertion
            solutions = np.reshape(solutions, (1, self._P))

        self._objvals = np.append(self._objvals, objvals)
        self._solutions = np.append(self._solutions, solutions, axis = 0)
        return self


    def filter(self, filter_ind):
        idx = np.require(filter_ind, dtype = 'bool').flatten()
        if len(self) > 0 and any(idx == 0):
            self._objvals = self._objvals[idx]
            self._solutions = self._solutions[idx, :]
        return self


    def distinct(self):
        if len(self) > 0:
            _, idx = np.unique(self._solutions, return_index = True, axis = 0)
            self._objvals = self._objvals[idx]
            self._solutions = self._solutions[idx, :]
        return self


    def sort(self):
        if len(self) > 0:
            idx = np.argsort(self._objvals)
            self._objvals = self._objvals[idx]
            self._solutions = self._solutions[idx, :]
        return self


    def map(self, mapfun, target = 'all'):
        assert callable(mapfun), 'map function must be callable'
        if target == 'solutions':
            return list(map(mapfun, self.solutions))
        elif target == 'objvals':
            return list(map(mapfun, self.objvals))
        elif target == 'all':
            return list(map(mapfun, self.objvals, self.solutions))
        else:
            raise ValueError('target must be either solutions, objvals, or all')


    @staticmethod
    def is_integral(solution):
        return np.all(solution == np.require(solution, dtype = 'int_'))


    def remove_nonintegral(self):
        return self.filter(list(map(self.is_integral, self.solutions)))


    def compute_objvals(self, get_objval):
        compute_idx = np.flatnonzero(np.isnan(self._objvals))
        self._objvals[compute_idx] = np.array(list(map(get_objval, self._solutions[compute_idx, :])))
        return self


    def remove_suboptimal(self, objval_cutoff):
        return self.filter(self.objvals <= objval_cutoff)


    def remove_infeasible(self, is_feasible):
        return self.filter(list(map(is_feasible, self.solutions)))


class FastSolutionPool(object):
    """
    Helper class used to store solutions to the risk slim optimization problem
    SolutionQueue designed to work faster than SolutionPool.
    It is primarily used by the callback functions in risk_slim
    """

    def __init__(self, P):
        self._P = int(P)
        self._objvals = np.empty(shape = 0)
        self._solutions = np.empty(shape = (0, P))


    def __len__(self):
        return len(self._objvals)

    @property
    def P(self):
        return self._P

    @property
    def objvals(self):
        return self._objvals

    @property
    def solutions(self):
        return self._solutions


    def add(self, new_objvals, new_solutions):
        if isinstance(new_objvals, (np.ndarray, list)):
            n = len(new_objvals)
            self._objvals = np.append(self._objvals, np.array(new_objvals).astype(dtype = np.float_).flatten())
        else:
            n = 1
            self._objvals = np.append(self._objvals, float(new_objvals))

        new_solutions = np.reshape(new_solutions, (n, self._P))
        self._solutions = np.append(self._solutions, new_solutions, axis = 0)


    def get_best_objval_and_solution(self):
        if len(self) > 0:
            idx = np.argmin(self._objvals)
            return float(self._objvals[idx]), np.copy(self._solutions[idx,])
        else:
            return np.empty(shape = 0), np.empty(shape = (0, self.P))


    def filter_sort_unique(self, max_objval = float('inf')):

        # filter
        if max_objval < float('inf'):
            good_idx = np.less_equal(self._objvals, max_objval)
            self._objvals = self._objvals[good_idx]
            self._solutions = self._solutions[good_idx,]

        if len(self._objvals) >= 2:
            _, unique_idx = np.unique(self._solutions, axis = 0, return_index = True)
            self._objvals = self._objvals[unique_idx]
            self._solutions = self._solutions[unique_idx,]

        if len(self._objvals) >= 2:
            sort_idx = np.argsort(self._objvals)
            self._objvals = self._objvals[sort_idx]
            self._solutions = self._solutions[sort_idx,]

        return self


    def clear(self):
        self._objvals = np.empty(shape = 0)
        self._solutions = np.empty(shape = (0, self._P))
        return self


    def table(self):
        x = pt.PrettyTable(align = 'r', float_format = '1.4', hrules=pt.ALL)
        x.add_column("objval", self._objvals.tolist())
        x.add_column("solution", list(map(self.solution_string, self._solutions)))
        return str(x)

    @staticmethod
    def solution_string(solution):
        solution_string = ''
        for j in range(len(solution)):
            if SolutionPool.is_integral(solution[j]):
                solution_string += ' ' + str(int(solution[j]))
            else:
                solution_string += (' %1.4f' % solution[j])
        return solution_string

    def __repr__(self):
        return self.table()


    def __str__(self):
        return self.table()
