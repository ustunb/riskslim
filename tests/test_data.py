"""Test data objects."""

import numpy as np
from riskslim.utils import Stats
from riskslim.bounds import Bounds
from riskslim.data import BinaryClassificationDataset


def test_bounds():

    bounds = Bounds(objval_min=0., objval_max=1., loss_min=0., loss_max=1.,
                    min_size=1, max_size=10)

    assert bounds.objval_min == bounds.loss_min == 0.
    assert bounds.objval_max == bounds.loss_max == 1.
    assert bounds.min_size == 1 and bounds.max_size == 10

    bounds = bounds.asdict()
    assert isinstance(bounds, dict)



def test_stats():

    incumbent = np.zeros(10)

    stats = Stats(incumbent)

    stats_dict = stats.asdict()
    assert isinstance(stats_dict, dict)

    stats = Stats(incumbent)

    for k in stats_dict.keys():
        val = getattr(stats, k)

        if k == 'incumbent':
            assert np.all(val == incumbent)
        elif k == 'bounds':
            assert isinstance(val, Bounds)
        else:
            assert hasattr(stats, k)
            assert isinstance(val, (int, float))
            assert (val == 0) or not np.isfinite(val)


def test_read_csv_without_data_suffix_infers_columns(tmp_path):
    # a file not named *_data.csv has no implied helper file, so column roles come from the data
    data_file = tmp_path / "x_binarized.csv"
    data_file.write_text("y,a,b\n" + "0,1,0\n1,0,1\n" * 5)

    data = BinaryClassificationDataset.read_csv(data_file)

    assert data.names.y == "y" and data.names.X == ["a", "b"]
