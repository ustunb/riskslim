def get_loss_functions(Z, loss_computation):
    """Initalize loss functions on the signed design matrix Z (C-contiguous float64)."""

    assert loss_computation in ("normal",)

    if loss_computation == "normal":
        from .loss_functions import log_loss as lf

        handles = {
            'loss': lambda rho: lf.log_loss_value(Z, rho),
            'loss_cut': lambda rho: lf.log_loss_value_and_slope(Z, rho),
            'loss_from_scores': lambda scores: lf.log_loss_value_from_scores(scores),
            }

        # add handles for real-valued losses
        handles_real = {k + '_real': f for k, f in handles.items()}
        handles.update(handles_real)

    return handles
