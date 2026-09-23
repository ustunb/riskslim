"""Dataset module for binary classification data."""

from .data import BinaryClassificationDataset, BinaryClassificationSample
from .groups import GroupInfo, GroupAttributeEncoder, EncodingType
from .binarizers import (
    Binarizer,
    NumericBinarizer,
    CategoricalBinarizer,
    BooleanBinarizer,
    ThresholdBinarizer,
    BinBinarizer,
    Ordinal,
    Categorical,
)
from .processor import Processor, GroupAttribute, Label, Feature
from .names import Names, RuleName
from .cv import CVFolds, FoldVector
from .pipeline import Pipeline

__all__ = [
    "BinaryClassificationDataset",
    "BinaryClassificationSample",
    "GroupInfo",
    "GroupAttributeEncoder",
    "EncodingType",
    "Binarizer",
    "NumericBinarizer",
    "CategoricalBinarizer",
    "BooleanBinarizer",
    "ThresholdBinarizer",
    "BinBinarizer",
    "Ordinal",
    "Categorical",
    "Processor",
    "GroupAttribute",
    "Label",
    "Feature",
    "Names",
    "RuleName",
    "CVFolds",
    "FoldVector",
    "Pipeline",
]

