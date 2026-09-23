from .loss_functions import log_loss as lf


def get_loss_functions(Z):
    """Log loss handles on the signed design matrix Z (C-contiguous float64)."""
    return {
        'loss': lambda weights: lf.log_loss_value(Z, weights),
        'loss_cut': lambda weights: lf.log_loss_value_and_slope(Z, weights),
        'loss_from_scores': lambda scores: lf.log_loss_value_from_scores(scores),
        }
