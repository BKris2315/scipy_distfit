import numpy as np
from distfit import distfit
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, Optional, Dict, List, Union

from src.pdfs import PDFs

def log_binning(data: np.ndarray, log_base: float = 1.12, max_bins: int = 100, 
                min_exponent: int = 1,
                max_exponent: int = 122) -> np.ndarray:
    """
    Perform logarithmic binning on a numpy array.
    
    Args:
        data (np.ndarray): Input data array.
        log_base (float, optional): Base of the logarithm for binning. Defaults to 1.12.
        max_bins (int, optional): Maximum number of bins. Defaults to 100.
        min_exponent (int, optional): Minimum exponent to generate bin edges. Defaults to 1.
        max_exponent (int, optional): Maximum exponent to generate bin edges. Defaults to 100.
    
    Returns:
        np.ndarray: A 2D array where the first column is the bin centers and the second column is the normalized count per bin.
    """
    bins = np.unique(log_base ** np.arange(min_exponent, max_exponent))[:max_bins+1]
    counts, _ = np.histogram(data, bins=bins)
    bin_widths = np.diff(bins)
    normalized_counts = counts / bin_widths
    norm_factor = normalized_counts.sum()
    normalized_counts /= norm_factor if norm_factor > 0 else 1
    
    return np.column_stack((bins[:-1], normalized_counts))
    
def pdf_fitter(data: np.ndarray,
               x_min: Optional[float] = None,
               x_max: Optional[float] = None,
               pdf_type: str = 'powerlaw',
               initial_guess: np.ndarray = None,
               bounds: Union[List[List[float]], np.ndarray] = None,
               max_fev: int = 10000) -> Tuple[str, Dict[str, float]]:
    """
    Fit a probability distribution to a specified range of the binned data.

    Args:
        data (np.ndarray): A 2D array where the first column contains x-values 
        and the second column contains the corresponding probabilities.
        x_min (float, optional): Lower bound of the fitting range. Defaults to None.
        x_max (float, optional): Upper bound of the fitting range. Defaults to None.
        pdf_type (str, optional): Name of the probability distribution to fit. Defaults to 'powerlaw'.
        initial_guess (np.ndarray, optional): Initial guess for the distribution parameters. Defaults to None.
        bounds (Union[List[float], np.ndarray], optional): Bounds for the distribution parameters. Defaults to None.
        max_fev (int, optional): Maximum number of function evaluations. Defaults to 10000.

    Returns:
        Tuple[str, Dict[str, float]]: Best-fitting distribution name and its parameters.
    """

    if not pdf_type in list(PDFs.__dict__.keys())[3:-3]:
        raise ValueError(f"pdf_type must be one of {list(PDFs.__dict__.keys())[3:-3]}")
    
    # Filter data within the specified range
    mask = np.ones(len(data), dtype=bool)
    if x_min is not None:
        mask &= data[:, 0] >= x_min 
    if x_max is not None:
        mask &= data[:, 0] <= x_max
    mask &= data[:, 1] > 0
    filtered_data = data[mask]

    distribution_func = getattr(PDFs, pdf_type)
    nr_of_params = distribution_func.__code__.co_argcount - 1

    try:
        if initial_guess is not None:
            if len(initial_guess) == nr_of_params:
                initial_guess = initial_guess
    except:
        raise ValueError(f"Number of initial guesses {len(initial_guess)} are not matching number of params {nr_of_params}.")


    if bounds is not None:
        bounds = np.array(bounds)

        if bounds.shape[0] != 2 or bounds.shape[1] != nr_of_params:
            raise ValueError(f"Bounds must be a list of shape (2, {nr_of_params}).")
    else:
        bounds = ([-np.inf]*nr_of_params, [np.inf]*nr_of_params)
    
    # Fit distributions
    popt, pcov = curve_fit(distribution_func, filtered_data[:,0], filtered_data[:,1], p0=initial_guess, bounds=bounds, maxfev=max_fev)
    opt_params = {key: val for key, val in zip(distribution_func.__code__.co_varnames[1:], popt)}
    return opt_params, pcov
