"""Array checks, encoding names and dill serialization shared by the dataset modules."""

from copy import copy
from enum import StrEnum
from pathlib import Path

import dill
import numpy as np


def is_boolean_array(X, encoding="any"):
    """Check if array contains only boolean values.

    :param X: Array to check.
    :param encoding: Which encoding to check for:
        - 'any': Either {0,1} or {-1,1} (default)
        - '01': Only {0,1}
        - 'pm1': Only {-1,1}
    :returns: True if all values match the specified encoding.
    """
    if encoding == "any":
        out = np.isin(X, (0, 1)).all() or np.isin(X, (-1, 1)).all()
    elif encoding == "01":
        out = np.isin(X, (0, 1)).all()
    elif encoding == "pm1":
        out = np.isin(X, (-1, 1)).all()
    else:
        raise ValueError(f"encoding must be 'any', '01', or 'pm1', got '{encoding}'")
    return out


class DillSerializableMixin:
    """Mixin for dill-based serialization with integrity checks.

    Requires the class to implement:
    - __eq__(): For save verification (if check_save=True).

    Optional:
    - __check_rep__(): Returns True if object is valid. Called on save/load if present.
    - reset(): Called on copy before saving.
    """

    def save(self, file, overwrite=False, check_save=True):
        """Save object to disk using dill.

        :param file: Path to save to.
        :param overwrite: If True, overwrite existing file.
        :param check_save: If True, verify save by loading and comparing.
        :return: Path to saved file.
        """
        f = Path(file)
        assert overwrite or not f.is_file(), f'file {f} already exists on disk'
        check_rep = getattr(self, '__check_rep__', None)
        if check_rep is not None:
            assert check_rep()

        data = copy(self)
        reset = getattr(data, 'reset', None)
        if reset is not None:
            reset()
        with open(f, 'wb') as outfile:
            dill.dump(data, outfile, protocol=dill.HIGHEST_PROTOCOL)

        if check_save:
            assert data == self.load(file=f)
        return f

    @classmethod
    def load(cls, file):
        """Load object from disk.

        :param file: Path to load from.
        :return: Loaded object.
        """
        f = Path(file)
        assert f.is_file(), f"file: {f} not found"
        with open(f, 'rb') as infile:
            data = dill.load(infile)
        check_rep = getattr(data, '__check_rep__', None)
        if check_rep is not None:
            assert check_rep(), 'loaded data has been corrupted'
        return data


class EncodingType(StrEnum):
    """Encoding types for group attributes."""
    ONEHOT = "onehot"
    INTERSECTIONAL = "intersectional"
