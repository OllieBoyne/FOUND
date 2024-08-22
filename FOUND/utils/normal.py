import numpy as np


def kappa_to_alpha_np(kappa: np.ndarray):
    """Compute angular uncertainty in degrees from kappa values"""
    alpha = ((2 * kappa) / ((kappa**2.0) + 1)) + (
        (np.exp(-kappa * np.pi) * np.pi) / (1 + np.exp(-kappa * np.pi))
    )
    alpha = np.degrees(alpha)
    return alpha
