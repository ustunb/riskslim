import numpy as np


def get_loss_functions(data, coef_set, loss_computation, max_size = None):
    """Initalize loss functions."""

    assert loss_computation in ("normal",)
    max_size = data.d if max_size is None else max_size

    if loss_computation == "normal":
        from .loss_functions import log_loss as lf
        data._Z = np.require(data._Z, requirements=["C"])

        handles = {
            'loss': lambda rho: lf.log_loss_value(data.Z, rho),
            'loss_cut': lambda rho: lf.log_loss_value_and_slope(data.Z, rho),
            'loss_from_scores': lambda scores: lf.log_loss_value_from_scores(scores),
            }

        # add handles for real-valued losses
        handles_real = {k + '_real': f for k, f in handles.items()}
        handles.update(handles_real)

    return handles
