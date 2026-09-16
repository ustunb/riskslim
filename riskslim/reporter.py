"""Risk scores results, measures, and reports."""

from pathlib import Path
import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.calibration import CalibratedClassifierCV

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import write_image

from riskslim.utils import print_model

REPORT_FILE_TYPES = ('.pdf', '.png', '.html')

class RiskScoreReporter:
    """Risk scores (rho), derived metrics, and reports."""

    def __init__(self, dataset, estimator):
        """Initalize object

        Parameters
        ----------
        dataset : risklim.data.ClassificationDataset
            Binary classification data.
        estimator : riskslim.classifier.RiskSLIMClassifier
            Fitted riskslim estimator. Also accepts scikit-learn estimators.
        """

        if not isinstance(estimator, list) and estimator.coef_ is None:
            # Ensure single model if fit
            raise ValueError("RiskScore expects a fit RiskSLIM or linear model input.")

        self.estimator = estimator
        self.X = dataset.X
        self.y = dataset.y
        self.variable_names = dataset.variable_names
        self.outcome_name = dataset.outcome_name

        if hasattr(estimator, "rho"):
            self.rho = estimator.rho
        else:
            # For scikit-learn estimators
            self.rho = np.insert(np.squeeze(self.estimator.coef_), 0, self.estimator.intercept_)
            self.variable_names.insert(0, "(Intercept)")

        if hasattr(estimator, "_variable_types"):
            self._variable_types = estimator._variable_types
        else:
            # For scikit-learn estimators
            self._variable_types = np.zeros(self.X.shape[1], dtype="str")
            self._variable_types[:] = "C"
            self._variable_types[np.all(self.X == np.require(self.X, dtype=np.int_), axis=0)] = "I"
            self._variable_types[np.all(self.X == np.require(self.X, dtype=np.bool_), axis=0)] = "B"

        # Table
        if np.not_equal(estimator.coef_, 0.0).any():
            self.table_str = print_model(
                self.rho,
                self.variable_names,
                self.outcome_name,
                show_omitted_variables=False,
                return_only=True
            )
            self.table = {}
            self._prepare_table()

        # Probability estimates
        if not hasattr(self.estimator, "calibrated_estimator") or self.estimator.calibrated_estimator is None:
            self.proba = estimator.predict_proba(self.X)
        else:
            self.proba = self.estimator.calibrated_estimator.predict_proba(self.X)[:, 1]

    @staticmethod
    def from_model(estimator):
        reporter = RiskScoreReporter(estimator._data, estimator)
        return reporter

    def __str__(self):
        """Scores string."""
        return str(self.table_str)


    def __repr__(self):
        return str(self.table_str)


    def print_coefs(self):
        """Print coefficient info."""
        if hasattr(self.estimator.coef_set):
            print(self.estimator.coef_set)
        else:
            print(self.estimator.coef_)


    def compute_metrics(self, y, proba, n_bins=5):
        """Computes calibration and ROC."""

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y, proba, pos_label=1, n_bins=n_bins)
        prob_pred *= 100
        prob_true *= 100

        # ROC curve
        fpr, tpr, _ = roc_curve(y.reshape(-1), proba)

        return prob_pred, prob_true, fpr, tpr


    def _prepare_table(self):
        """Prepare arrays for plotly table."""

        # Non-zero coefficients
        inds = np.flatnonzero(self.rho[1:])
        if len(inds) == 0:
            raise ValueError('all zero coefficients')

        self.table["names"] = np.array(self.variable_names)[inds+1].tolist()
        self.table["scores"] = self.rho[inds+1]

        self.table["names"] = [str(i+1) + '.    ' + n
                             for i, n in enumerate(self.table["names"])]
        self.table["names"].append('')

        self.table["scores"] = ['(' + str(int(s)) + ' point)' if s == 1 else str(int(s)) + ' points)'
                              for s in self.table["scores"]]
        self.table["scores"].append('SCORE')

        self.table["last_col"] = ['   ...', *['+ ...']*(len(self.table["scores"])-2)]
        self.table["last_col"].append('= ...')


    def create_report(self, file_name=None, show=False, replace_table=False,
                      only_table=False, template=None, n_bins=5, overwrite=True):
        """Create a RiskSLIM report using plotly.

        Parameters
        ----------
        file_name : str
            Name of file and extension to save report to.
            Supported extensions include ".pdf" and ".html".
        show : bool, optional, default: True
            Calls fig.show() if True.
        replace_table : bool, optional, default: False
            Removes risk score table if True.
        only_table : bool, optional, default: False
            Plots only the risk table when True.
        template : str
            Path to html file template that will overwrite default.
        n_bins : int
            Number of to use when creating calibration plot.
        """
        # Overwrite
        if file_name is not None:
            file_name = Path(file_name)
            if not overwrite:
                assert file_name.exists(), f'File: {file_name} exists.'

        # Determine size of plots and tables
        height = 800
        width = 800

        if replace_table and not only_table:
            height = 600
        elif replace_table and only_table:
            height = 400
        elif only_table:
            height = 300

        # Paths
        if template is None:
            report_template = Path(__file__).parent / 'template.html'

        if not report_template.is_file():
            raise ValueError("Report template file does not exist.")

        write_file = file_name is not None

        if write_file:
            assert file_name.suffix in REPORT_FILE_TYPES

        if write_file and file_name.suffix in (".html"):
            # Generate figure html string
            fig = self.create_report(replace_table=True, show=False, n_bins=n_bins)
            fig.update_layout(font_family="Source Code Pro")
            fig_str = fig.to_html(include_plotlyjs=False, full_html=False)

            inds = np.flatnonzero(self.rho)
            _vars = np.array(self.variable_names)[inds]
            _rho = self.rho[inds]

            min_values = self.X.min(axis=0)[inds]
            max_values = self.X.max(axis=0)[inds]

            # Read html
            with open(report_template, "r") as f:
                text = f.read()
            text_list = text.split("\n")

            # Inject variable names & rho into js
            for ind, line in enumerate(text_list):
                if line.startswith("    var variable_names ="):
                    text_list[ind] = "    var variable_names = ["
                    for v in _vars:
                        text_list[ind] += f"\"{v}\","
                    text_list[ind] += "];"
                elif line.startswith("    var rho ="):
                    text_list[ind] = "    var rho = ["
                    for v in _rho:
                        text_list[ind] += f"{int(v)},"
                    text_list[ind] += "];"
                elif line.startswith("    var min_values ="):
                    text_list[ind] = "    var min_values = ["
                    for v in min_values:
                        text_list[ind] += f"{v},"
                    text_list[ind] += "];"
                elif line.startswith("    var max_values ="):
                    text_list[ind] = "    var max_values = ["
                    for v in max_values:
                        text_list[ind] += f"{v},"
                    text_list[ind] += "];"
                elif line.startswith("    var variable_types ="):
                    text_list[ind] = "    var variable_types = ["
                    for v in self._variable_types:
                        text_list[ind] += f"\"{v}\","
                    text_list[ind] += "];"
                elif line.startswith("        $ploty_html") and not only_table:
                    text_list[ind] = "    " + fig_str
                elif line.startswith("        $ploty_html") and only_table:
                    text_list[ind] = ""

            text = "\n".join(text_list)

            # Write html
            with open(file_name, "w") as f:
                f.write(text)

            return

        # Probabilties and postive rates
        self.prob_pred, self.prob_true, self.fpr, self.tpr = \
            self.compute_metrics(self.y, self.proba, n_bins=n_bins)

        # Initalize subplots
        row = 1
        if not replace_table and not only_table:
            fig = make_subplots(
                rows=3, cols=2,
                specs=[
                    [{"colspan": 2, "type": "table"}, None],
                    [{"colspan": 2, "type": "table"}, None],
                    [{}, {}],
                ],
                row_heights=[.3, .2, .5],
                vertical_spacing=.05,
                subplot_titles=("Scores", "Predicted Risk", "Calibration", "ROC Curve")
            )
        elif not replace_table and only_table:
            fig = go.Figure()
            fig.update_layout({"title": "Score"})
        else:
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"colspan": 2, "type": "table"}, None],
                    [{}, {}],
                ],
                row_heights=[.3, .7],
                vertical_spacing=.05,
                subplot_titles=("Predicted Risk", "Calibration", "ROC Curve")
            )

        # Table: risk scores
        if not replace_table:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=[''],
                        height=0
                    ),
                    cells=dict(
                        values=[
                            [*self.table["names"]],
                            [*self.table["scores"]],
                            [*self.table["last_col"]]
                        ],
                        align='left',
                    ),
                    columnwidth=[400, 150, 150]
                ),
                row=row if not only_table else None,
                col=1 if not only_table else None
            )
            row += 1

        if not only_table:
            # Table: predicted risk
            _pred_risk = np.stack((
                np.arange(len(self.prob_pred)),
                [str(round(i, 1))+"%" for i in self.prob_pred.tolist()]
            )).T

            fig.add_trace(
                go.Table(
                    header=dict(
                        values=[''],
                        height=0
                    ),
                    cells=dict(
                        values=[
                            ['Score', 'Risk'],
                            *_pred_risk
                        ],
                        align='left',
                    ),
                ),
                row=row,
                col=1
            )

            # Folds
            if hasattr(self.estimator, "cv_results") and self.estimator.cv_results is not None:

                for ind, (_, test) in enumerate(self.estimator.cv.split(self.X)):

                    # Calibration
                    if self.estimator.cv_calibrated_estimators_ is None:
                        prob_pred, prob_true, fpr, tpr = self.compute_metrics(
                            self.y[test],
                            self.estimator.cv_results["estimator"][ind].predict_proba(self.X[test]),
                            n_bins=n_bins
                        )
                    else:
                        prob_pred, prob_true, fpr, tpr = self.compute_metrics(
                            self.y[test],
                            self.estimator.cv_calibrated_estimators_[ind].predict_proba(self.X[test])[:, 1],
                            n_bins=n_bins
                        )

                    fig.add_trace(
                        go.Scattergl(
                            x=prob_pred,
                            y=prob_true,
                            mode='markers+lines',
                            line=dict(color='black', width=2),
                            opacity=.2,
                            name=""
                        ),
                        row=row+1,
                        col=1
                    )

                    # ROC
                    fig.add_trace(
                        go.Scattergl(
                            x=fpr,
                            y=tpr,
                            mode='markers+lines',
                            line=dict(color='black', width=2),
                            name="",
                            opacity=.2
                        ),
                        row=row+1,
                        col=2
                    )

            # Reliability Diagram
            fig.add_trace(
                go.Scattergl(
                    x=self.prob_pred,
                    y=self.prob_true,
                    mode='markers+lines',
                    line=dict(color='black', width=2),
                    name="Model"
                ),
                row=row+1,
                col=1
            )

            fig.add_trace(
                go.Scattergl(
                    x=np.linspace(0, 100),
                    y=np.linspace(0, 100),
                    mode='lines',
                    line=dict(color='black', dash='dash', width=2),
                    opacity=.5,
                    name="Ideal"
                ),
                row=row+1,
                col=1
            )


            # ROC Curve
            fig.add_trace(
                go.Scattergl(
                    x=self.fpr,
                    y=self.tpr,
                    mode='markers+lines',
                    line=dict(color='black', width=2),
                    name="ROC"
                ),
                row=row+1,
                col=2
            )

            fig.add_trace(
                go.Scattergl(
                    x=np.linspace(0, 1),
                    y=np.linspace(0, 1),
                    mode='lines',
                    line=dict(color='black', dash='dash', width=2),
                    opacity=.5,
                    name="Chance"
                ),
                row=row+1,
                col=2
            )

        # Update Attributes
        fig.update_layout(
            # General
            title_text="RiskSLIM Report" if not replace_table and not only_table else "",
            autosize=False,
            width=width,
            height=height,
            showlegend=False,
            template='simple_white',
            # Calibration axes
            yaxis1=dict(
                tickmode='array',
                tickvals=np.arange(0, 120, 20),
                ticktext=[str(i) + '%' for i in np.arange(0, 120, 20)],
                title='Observed Risk'
            ) if not only_table else None,
            xaxis1=dict(
                tickmode='array',
                tickvals=np.arange(0, 120, 20),
                ticktext=[str(i) + '%' for i in np.arange(0, 120, 20)],
                title='Predicted Risk'
            ) if not only_table else None,
            xaxis2=dict(
                range=(-.05, 1),
                tickmode='array',
                tickvals=np.linspace(0, 1, 6),
                title='False Positive Rate'

            ) if not only_table else None,
            yaxis2=dict(
                tickmode='array',
                tickvals=np.linspace(0, 1, 6),
                title='True Positive Rate'
            ) if not only_table else None
        )

        if write_file:
            if file_name.suffix in ('.pdf', '.png'):
                write_image(fig, file_name, format=file_name.suffix[1:])
            elif file_name.suffix in ('.html'):
                with open(file_name, 'w') as f:
                    f.write(fig.to_html())

        if show:
            fig.show()

        return fig
