import typing as tp

import numpy as np


class Scaler:
    """Scaler class according to LLMTime paper. Note that this scaler should be fitted on the Training dataset, as
    otherwise knowledge is leaked.
    Args:
        transform (Callabel): Function to map array-like object to transformed representation.
        inverse_transform (Callable): Function to apply inverse to array-like to re-construct values.
    """

    def __init__(self, transform=lambda x: x, inverse_transform=lambda x: x):
        self.transform: tp.Callable[[np.array], np.array] = transform
        self.inverse_transform: tp.Callable[[np.array], np.array] = inverse_transform


def get_scaler(time_series: np.array, quantile: float = 0.95, beta: float = 0.3, default: bool = False):
    """
    Generate a Scaler object based on given history data using curried application of scaling functions.

    Args:
        time_series (np.array): Time-series to fit transformation to.
        quantile (float, default=0.95): Quantile (alpha in paper) [0, 1) used for scaling.
        beta (float, default=0.3): Shift for scaling parameter.
        default (bool, default=False): If True, no shift is applied, and scaling is clipped to 0.01.

    Returns:
        Scaler:  scaler object.
    """
    if default:
        # Limit quantile scaling to 0.01, to prevent `float('inf')`
        local_q = max(np.quantile(np.abs(time_series), q=quantile).item(), 0.01)

        def transform(x):
            return x / local_q

        def inverse_transform(x):
            return x * local_q

    else:
        local_min = np.min(time_series) - beta * (np.max(time_series) - np.min(time_series))
        local_q = np.quantile(time_series - local_min, q=quantile)
        # Limit scaling to 0.01, to prevent `float('inf')`
        if abs(local_q) < 0:
            local_q = np.sign(local_q) * 0.01

        if local_q == 0:
            local_q = 1
        def transform(x):
            return (x - local_min) / local_q
        def inverse_transform(x):
            return x * local_q + local_min

    return Scaler(transform=transform, inverse_transform=inverse_transform)
