import numpy as np
import scipy  # type: ignore
from scipy.stats import gaussian_kde, norm  # type: ignore

def ll_samples(y, y_hat, weights=None, return_average=True):
    """Partially copied from: https://github.com/toonvds/NOFLITE/blob/main/metrics.py"""
    # region_size = 0.5
    # EPS = 1e-12
    if weights is not None:
        weights = weights / np.sum(weights, axis=1)[:, None]
    lls = np.zeros(len(y))

    for i in range(len(y)):
        # TODO: check if this is correct, you get a problem here when kde is impossible to fit
        # Already made an adjustment for this like returning nan if one of the kde's fails,
        # and also return on average nan if one of the samples fails
        try:
            if weights is not None:
                kde = gaussian_kde(dataset=y_hat[i, :], weights=weights[i, :])
            else:
                kde = gaussian_kde(dataset=y_hat[i, :])
            ll = kde.logpdf(y[i])
        except Exception as e:
            print(f"Error in loglikelihood calculation for sample {i}: {e}")
            ll = np.nan
        lls[i] = ll

    if return_average:
        if np.isnan(lls).any():
            return np.nan
        return np.mean(lls)
    else:
        return lls


def crps(y, cdf, return_average=True):
    """Calculate the discrete Continuous Ranked Probability Score (CRPS)

    Args:
        y: The true values (n,)
        cdf: The discrete cumulative distribution function (CDF) of the forecasted distribution (n, m, 2)
            where m is the number of discrete points in the CDF and in the last dimension in first index
            the CDF inputs and in the second index the CDF valeus (i.e, corresponding quantiles).
        return_average (bool, optional): If True, return the average CRPS. If False, return CRPS for each sample. Defaults to True.
    """
    # Check if cdf is a list of arrays
    if isinstance(cdf, list):
        results = []
        for i, cdf_i in enumerate(cdf):
            widths = np.hstack((cdf_i[0, 1], np.diff(cdf_i[:, 1])))
            result = 2 * np.sum(
                widths * ((y[i] < cdf_i[:,0]) - cdf_i[:, 1] + 0.5 * widths) * (cdf_i[:,0] - y[i])
            )
            results.append(result)
        if return_average:
            return np.mean(results)
        return np.array(results)
    # Ensure y is a column vector for broadcasting
    y = y[:, None]

    # Extract CDF values and calculate bin widths
    cdf_input = cdf[:, :, 0]  # y_hat values
    cdf_values = cdf[:, :, 1]  # F(y_hat) values

    widths = np.diff(cdf_values)
    widths = np.hstack((cdf_values[:, 0, None], widths))  # Prepend the first value to widths

    # Calculate CRPS
    crps_results = 2 * np.sum(
        widths * ((y < cdf_input) - cdf_values + 0.5 * widths) * (cdf_input - y), axis=1
    )

    if return_average:
        return np.mean(crps_results)
    return crps_results


def crps_samples(y, y_hat, return_average=True):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for sample forecasts.

    CRPS(F, y) = E_F |Y - y| - 0.5 E_F |Y - Y'|, where Y ~ F and Y' ~ F.

    :y: The true values (n,)
    :y_hat: The predicted values (n, samples)
    :return_average: If True, return the average CRPS over all samples. If False, return CRPS for each sample.
    """
    # Ensure y is a column vector for broadcasting
    y = y[:, None]

    # Compute the first term: E_F |Y - y|
    term1 = np.mean(np.abs(y_hat - y), axis=1)

    # Compute the second term: 0.5 * E_F |Y - Y'|
    term2 = np.zeros(len(y))
    for i in range(len(y)):
        # Compute all pairwise differences for the i-th observation's samples
        diff = y_hat[i, :, None] - y_hat[i, :]  # Shape (samples, samples)
        # Take the mean of all absolute pairwise differences
        term2[i] = np.mean(np.abs(diff))
    term2 *= 0.5  # Apply the 0.5 factor once after the loop

    # Compute CRPS for each sample
    crps_results = term1 - term2

    if return_average:
        return np.mean(crps_results)
    else:
        return crps_results


def crps_normal(y, mu, sigma, return_average=True):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a normal distribution.

    :y: The true values (n,)
    :mu: The mean of the normal distribution (n,)
    :sigma: The standard deviation of the normal distribution (n,)
    """
    # Ensure y is a column vector for broadcasting
    w = (y - mu) / sigma
    crps = sigma * (
        w * (2 * scipy.stats.norm.cdf(w) - 1) + 2 * scipy.stats.norm.pdf(w) - 1 / np.sqrt(np.pi)
    )
    if return_average:
        return np.mean(crps)
    return crps


def crps_weighted(y, y_hat, weights, return_average=True, batch_size=1000):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) using weighted forecasts.

    Parameters:
    y (np.array): Observed values, shape (n_samples,)
    y_hat (np.array): Forecasted values, shape (n_samples, n_forecast)
    weights (np.array): Weights for each forecast, shape (n_samples, n_forecast)
    return_average (bool): If True, return the mean CRPS. Otherwise, return individual scores.
    batch_size (int): Batch size for processing to manage memory usage.

    Returns:
    float or np.array: CRPS score(s)
    """
    # Normalize weights to sum to 1 for each sample
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    y = y[:, None]  # Reshape y to (n_samples, 1)

    # Compute term1: sum(weights * |y_hat - y|)
    term1 = np.sum(weights * np.abs(y_hat - y), axis=1)

    term2 = np.zeros(len(y))
    n_forecast, n_samples = y_hat.shape

    if n_forecast > n_samples:
        # Process samples in batches
        for i in range(0, n_forecast, batch_size):
            end_idx = min(i + batch_size, n_forecast)
            batch_y_hat = y_hat[i:end_idx]
            batch_weights = weights[i:end_idx]
            diff = np.abs(batch_y_hat[:, :, None] - batch_y_hat[:, None, :])
            weight_prod = batch_weights[:, :, None] * batch_weights[:, None, :]
            term2[i:end_idx] = np.sum(diff * weight_prod, axis=(1, 2))
    else:
        # Process forecast points in batches with nested loops
        cum_weights = np.cumsum(weights, axis=1)
        cum_y = np.cumsum(weights * y_hat, axis=1)

        cum_weights_shifted = np.pad(cum_weights[:, :-1], ((0, 0), (1, 0)), mode="constant")
        cum_y_shifted = np.pad(cum_y[:, :-1], ((0, 0), (1, 0)), mode="constant")

        term2 = 2 * np.sum(
            y_hat * weights * cum_weights_shifted - cum_y_shifted * weights, axis=1
        )  # Times 2 for acounting for both sides of the distribution
    term2 *= 0.5
    crps_results = term1 - term2

    return np.mean(crps_results) if return_average else crps_results


def calculate_dispersion(y, dist, return_p_values=False):
    """
    Calculate the dispersion metric for a distribution.

    Parameters:
    y (np.array): Observed values, shape (n_samples,)
    dist (np.array): Forecasted sampled distribution, shape (n_samples, n_forecast)

    Returns:
    float: Dispersion metric
    """
    # Vectorized calculation of PIT values
    p_values = np.mean(dist <= y[:, None], axis=1)

    # Compute dispersion metric
    if return_p_values:
        return np.var(p_values), p_values
    return np.var(p_values)


def calculate_dispersion_normal(y, mu, sigma, return_p_values=False):
    """
    Calculate the dispersion metric assuming a normal distribution.

    Parameters:
    y (np.array): Observed values, shape (n_samples,)
    mu (np.array): Mean predictions, shape (n_samples,)
    sigma (np.array): Standard deviations, shape (n_samples,)

    Returns:
    float: Dispersion metric
    """
    # Compute PIT values assuming a normal distribution
    p_values = norm.cdf(y, loc=mu, scale=sigma)

    # Compute dispersion metric
    if return_p_values:
        return np.var(p_values), p_values
    return np.var(p_values)


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute the Brier score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
        The Brier score.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_prob, np.ndarray):
        raise ValueError("y_true and y_prob must be numpy arrays")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("y_prob must be in the range [0, 1]")

    return np.mean((y_true - y_prob) ** 2)


def log_score(y_true, y_prob):
    """
    Compute the logarithmic score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
        The logarithmic score.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_prob, np.ndarray):
        raise ValueError("y_true and y_prob must be numpy arrays")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("y_prob must be in the range [0, 1]")

    return -np.mean(
        np.log(np.clip(y_prob, 1e-15, 1 - 1e-15)) * y_true
        + np.log(np.clip(1 - y_prob, 1e-15, 1 - 1e-15)) * (1 - y_true)
    )


def misclassification_score(y_true, y_prob):
    """
    Compute the misclassification score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
        The misclassification score.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_prob, np.ndarray):
        raise ValueError("y_true and y_prob must be numpy arrays")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("y_prob must be in the range [0, 1]")

    return np.mean(
        ((y_prob < 0.5) * y_true).astype(int)
        + ((y_prob > 0.5) * (1 - y_true)).astype(int)
        + 0.5 * (y_prob == 0.5).astype(int)
    )


def score_decomposition(y_true, y_prob, y_cal, y_ref=None, score="brier"):
    """
    Compute the CORP score decomposition (MCB, DSC, and UNC components).

    S = MCB - DSC + UNC
    where:
    - MCB=(S_{y_prob}-S_{y_cal}): The miscalibration component is the difference of the mean scores of the orignal and calibrated forecast.
    - DSC=(S_{marginal}-S_{cal}): The discrimination component quantifies the discrimination ability via the difference of the mean scores of the marginal/reference and calibrated forecast.
    - UNC=S_{marginal}: The uncertainty component is the mean score of the marginal/reference forecast.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities.
    y_cal : np.ndarray
        Calibrated probabilities.
    y_ref : np.ndarray or float, optional
        Reference probabilities for the marginal forecast. If None, it is assumed that the marginal forecast is the marginal event of the true labels.
    score : str, optional
        The scoring rule to use. Default is "brier". Other options can be "log" for logarithmic score or "misclassification" for misclassification error.

    Returns
    -------
    dict
        A dictionary containing:
        - 'mcb': Miscalibration component.
        - 'dsc': Discrimination component.
        - 'unc': Uncertainty component.
    """
    if (
        not isinstance(y_true, np.ndarray)
        or not isinstance(y_prob, np.ndarray)
        or not isinstance(y_cal, np.ndarray)
    ):
        raise ValueError("y_true, y_prob, and y_cal must be numpy arrays")
    if len(y_true) != len(y_prob) or len(y_true) != len(y_cal):
        raise ValueError("y_true, y_prob, and y_cal must have the same length")
    if np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("y_prob and y_cal must be in the range [0, 1]")
    if y_ref is not None:
        if not isinstance(y_ref, (np.ndarray, float)):
            raise ValueError("y_ref must be a numpy array or a float")
        if isinstance(y_ref, np.ndarray) and len(y_ref) != len(y_true):
            raise ValueError("y_ref must have the same length as y_true")
        if isinstance(y_ref, float):
            y_ref = np.full_like(y_true, y_ref)
    else:
        y_ref = np.mean(y_true)
        y_ref = np.full_like(y_true, y_ref)
    if score == "brier":
        score_func = brier
    elif score == "log":
        score_func = log_score
    elif score == "misclassification":
        score_func = misclassification_score
    else:
        raise ValueError(
            f"Unsupported score: {score}. Supported scores are 'brier', 'log', and 'misclassification'."
        )
    score_prob = score_func(y_true, y_prob)
    score_cal = score_func(y_true, y_cal)
    score_ref = score_func(y_true, y_ref)
    mcb = score_prob - score_cal
    dsc = score_ref - score_cal
    unc = score_ref
    return {"mcb": mcb, "dsc": dsc, "unc": unc}
