"""Report layout: rows of components, set in Python and drawn by one generic renderer in JS.

Adding a component means one dataclass here, one entry in the ``draw_report_components.js``
registry and, if it needs data, one function in ``build_report_data.py``.
"""

from dataclasses import asdict, dataclass

SUMMARY_BLOCKS = ("dataset", "constraints", "training", "performance")


@dataclass(frozen=True)
class ModelCard:
    """The model: a points table (risk score) or a list of items (checklist)."""

    title: str = "Model"


@dataclass(frozen=True)
class SummaryTable:
    """Summary blocks, shown in the order given."""

    title: str = "Summary"
    blocks: tuple = SUMMARY_BLOCKS

    def __post_init__(self):
        unknown = [b for b in self.blocks if b not in SUMMARY_BLOCKS]
        if unknown:
            raise ValueError(f"SummaryTable blocks must be among {SUMMARY_BLOCKS}; got {unknown}")
        object.__setattr__(self, "blocks", tuple(self.blocks))


@dataclass(frozen=True)
class RocPlot:
    """ROC curve per sample; the metrics box shows AUC."""

    title: str = "ROC"


@dataclass(frozen=True)
class CalibrationPlot:
    """Calibration bubbles per sample; the metrics box shows calibration error."""

    title: str = "Calibration"


COMPONENTS = (ModelCard, SummaryTable, RocPlot, CalibrationPlot)


@dataclass(frozen=True, init=False)
class Row:
    """One row of the report grid: ``Row(ModelCard(), SummaryTable())``."""

    components: tuple

    def __init__(self, *components):
        for c in components:
            if not isinstance(c, COMPONENTS):
                names = ", ".join(cls.__name__ for cls in COMPONENTS)
                raise TypeError(f"Row takes component specs ({names}); got {c!r}")
        object.__setattr__(self, "components", tuple(components))


def default_layout(model_type):
    """The default riskslim layout (the same 2x2 grid for both model types)."""
    if model_type not in ("risk_score", "checklist"):
        raise ValueError(f"model_type must be 'risk_score' or 'checklist'; got {model_type!r}")
    return [Row(ModelCard(), SummaryTable()), Row(RocPlot(), CalibrationPlot())]


def layout_to_json(layout):
    """``[{"components": [{"component": name, "options": {...}}]}]`` for the data block."""
    if isinstance(layout, Row) or not all(isinstance(row, Row) for row in layout):
        raise TypeError(f"layout must be a list of Row objects; got {layout!r}")
    return [{"components": [{"component": type(c).__name__,
                             "options": {k: list(v) if isinstance(v, tuple) else v
                                         for k, v in asdict(c).items()}}
                            for c in row.components]}
            for row in layout]
