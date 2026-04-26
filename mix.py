from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, hypsecant, t as student_t
from scipy.special import logsumexp
from scipy.optimize import root, minimize

def normal_pdf(x, mu, sd):
    """Univariate normal pdf."""
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))

def normal_mix_variates_2_components(p, nobs, m0=0.0, sd0=1.0, m1=0.0, sd1=1.0):
    """ Generate random variates from a finite mixture of two normal distributions. """
    # Choose a component for each step based on the probability p
    components = np.random.choice([0, 1], size=nobs, p=[p, 1-p])
    # Draw increments from the appropriate normal distribution for each step
    xran = np.where(components == 0, np.random.normal(m0, sd0, nobs), np.random.normal(m1, sd1, nobs))
    return xran

def fit_normal_mix_1d(x, n_components):
    """ fit a mixture of normals to univariate data """
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(x.reshape(-1, 1))  # Reshape data to a 2D array as expected by scikit-learn
    return {"weight":gmm.weights_,
            "mean":gmm.means_.flatten(),
            "sd":np.sqrt(gmm.covariances_.flatten())}

def print_fits_normal_mix_1d(x, min_components=1, max_components=3, end=None):
    """ fit normal mixtures to 1D data, print estimated parameters """
    for n_components in range(min_components, max_components+1):
            mix_fit = fit_normal_mix_1d(x, n_components)
            print("\n",pd.DataFrame(mix_fit).sort_values("weight",
                ascending=False).to_string(index=False), sep="")
    if end:
        print(end=end)

def find_best_aic_bic_fits_normal_mix(data, min_components=1,
    max_components=3, print_all_ic=False):
    """ find the best AIC and BIC fits of normal mixtures """
    best_aic = np.inf
    best_bic = np.inf
    best_gmm_aic = None
    best_gmm_bic = None
    for n_components in range(min_components, max_components+1):
        # Fit a Gaussian mixture with n components
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(data)
        # Calculate the AIC and BIC
        current_aic = gmm.aic(data)
        current_bic = gmm.bic(data)
        if print_all_ic:
            print("#components, AIC, BIC:",
                n_components, current_aic, current_bic)
        # Check if this model has a better AIC than the current best
        if current_aic < best_aic:
            best_aic = current_aic
            best_gmm_aic = gmm
        # Check if this model has a better BIC than the current best
        if current_bic < best_bic:
            best_bic = current_bic
            best_gmm_bic = gmm
    return best_gmm_aic, best_gmm_bic

def print_fits_normal_mix_aic_bic(x, min_components=1, max_components=3,
    print_all_ic=False):
    """ estimate and print the best AIC and BIC fits of normal mixtures to
    matrix x """
    best_gmm_aic, best_gmm_bic = find_best_aic_bic_fits_normal_mix(x,
        min_components, max_components, print_all_ic=print_all_ic)
    # Print the parameters of the best AIC fit
    print(f'\nBest model by AIC has {best_gmm_aic.n_components} components:')
    print(f'  Weights: {best_gmm_aic.weights_}')
    print(f'  Means: {best_gmm_aic.means_.ravel()}')
    print(f'  Covariances: {best_gmm_aic.covariances_.ravel()}')
    # Print the parameters of the best BIC fit
    print(f'\nBest model by BIC has {best_gmm_bic.n_components} components:')
    print(f'  Weights: {best_gmm_bic.weights_}')
    print(f'  Means: {best_gmm_bic.means_.ravel()}')
    print(f'  Covariances: {best_gmm_bic.covariances_.ravel()}')

def normal_mix_stats(wgt, xmean=None, xsd=None):
    """
    Compute the expected mean, standard deviation, skewness, and excess kurtosis of a normal mixture distribution.

    Parameters:
    wgt : 1D numpy array
        Weights of the normal mixture components.
    xmean : 1D numpy array or None, optional
        Means of the normal mixture components. If None, all means are set to 0.
    xsd : 1D numpy array or None, optional
        Standard deviations of the normal mixture components. If None, all standard deviations are set to 1.

    Returns:
    stats : dict
        Dictionary containing the mean, standard deviation, skewness, and kurtosis of the mixture.
    """
    n_components = len(wgt)
    if xmean is None:
        xmean = np.zeros(n_components)
    if xsd is None:
        xsd = np.ones(n_components)
    assert len(wgt) == len(xmean) == len(xsd), "Arrays wgt, xmean, and xsd must have the same length"
    wgt = wgt / np.sum(wgt)
    mean = np.sum(wgt * xmean)
    variance = np.sum(wgt * (xsd**2 + (xmean - mean)**2))
    std_dev = np.sqrt(variance)
    third_central_moment = np.sum(wgt * ((xmean - mean)**3 + 3 * (xmean - mean) * xsd**2))
    skewness = third_central_moment / std_dev**3
    fourth_central_moment = np.sum(wgt * ((xmean - mean)**4 + 6 * (xmean - mean)**2 * xsd**2 + 3 * xsd**4))
    kurtosis = fourth_central_moment / std_dev**4 - 3  # Excess kurtosis

    return {
        'mean': mean,
        'sd': std_dev,
        'skew': skewness,
        'kurtosis': kurtosis
    }

def sim_normal_mix(wgt, xmean=None, xsd=None, size=1):
    """
    Simulate random samples from a normal mixture distribution.

    Parameters:
    wgt : 1D numpy array
        Weights of the normal mixture components.
    xmean : 1D numpy array or None, optional
        Means of the normal mixture components. If None, all means are set to 0.
    xsd : 1D numpy array or None, optional
        Standard deviations of the normal mixture components. If None, all standard deviations are set to 1.
    size : int, optional
        Number of samples to generate.

    Returns:
    samples : 1D numpy array
        Array of simulated samples from the mixture distribution.
    """
    n_components = len(wgt)

    # Set xmean to zeros if None
    if xmean is None:
        xmean = np.zeros(n_components)

    # Set xsd to ones if None
    if xsd is None:
        xsd = np.ones(n_components)

    # Ensure that wgt, xmean, and xsd have the same size
    assert len(wgt) == len(xmean) == len(xsd), "Arrays wgt, xmean, and xsd must have the same length"

    # Normalize the weights so they sum to 1
    wgt = wgt / np.sum(wgt)

    # Generate component indices according to the weights
    components = np.random.choice(n_components, size=size, p=wgt)

    # Generate samples from the selected components
    samples = np.random.normal(loc=xmean[components], scale=xsd[components])
    return samples

def normal_mix_std(weights, sigmas):
    """
    Compute the standard deviation of a mixture of zero-mean normal distributions.

    Parameters:
    ----------
    weights : array-like, shape (n,)
        Weights of the Gaussian mixture components. Must sum to 1 and be non-negative.

    sigmas : array-like, shape (n,)
        Standard deviations of the Gaussian mixture components. Must be positive.

    Returns:
    -------
    float
        The standard deviation of the Gaussian mixture.

    Raises:
    ------
    ValueError
        If weights do not sum to 1, contain negative values, or sigmas are non-positive.
    """
    weights = np.array(weights)
    sigmas = np.array(sigmas)

    # Input validation
    if not np.all(weights >= 0):
        raise ValueError("All mixture weights must be non-negative.")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-8):
        raise ValueError("Mixture weights must sum to 1.")
    if not np.all(sigmas > 0):
        raise ValueError("All standard deviations (sigmas) must be positive.")

    # Compute variance of the mixture
    variance = np.sum(weights * sigmas ** 2)

    # Standard deviation is the square root of variance
    std_dev = np.sqrt(variance)

    return std_dev

def normal_mix_kurt(weights, sigmas):
    """
    Compute the excess kurtosis of a mixture of zero-mean normal distributions.

    Parameters:
    ----------
    weights : array-like, shape (n,)
        Weights of the Gaussian mixture components. Must sum to 1 and be non-negative.

    sigmas : array-like, shape (n,)
        Standard deviations of the Gaussian mixture components. Must be positive.

    Returns:
    -------
    float
        The excess kurtosis of the Gaussian mixture.

    Raises:
    ------
    ValueError
        If weights do not sum to 1, contain negative values, or sigmas are non-positive.
    """
    weights = np.array(weights)
    sigmas = np.array(sigmas)

    # Input validation
    if not np.all(weights >= 0):
        raise ValueError("All mixture weights must be non-negative.")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-8):
        raise ValueError("Mixture weights must sum to 1.")
    if not np.all(sigmas > 0):
        raise ValueError("All standard deviations (sigmas) must be positive.")

    # Compute variance of the mixture
    variance = np.sum(weights * sigmas ** 2)

    # Compute fourth moment of the mixture
    fourth_moment = np.sum(weights * 3 * sigmas ** 4)

    # Compute kurtosis
    kurtosis = fourth_moment / (variance ** 2)

    # Excess kurtosis is kurtosis minus 3
    excess_kurtosis = kurtosis - 3
    return excess_kurtosis

def normal_mix_variates(n, weights, means=None, sigmas=None):
    """
    Simulate n samples from a univariate normal mixture with specified weights, means, and sigmas.

    Parameters:
    ----------
    n : int
        Number of samples to generate. Must be a positive integer.

    weights : array-like, shape (k,)
        Weights of the Gaussian mixture components. Must be non-negative and sum to 1.

    means : array-like, shape (k,), optional
        Means of the Gaussian mixture components. If None, zero means are assumed.

    sigmas : array-like, shape (k,), optional
        Standard deviations of the Gaussian mixture components. If None, sigmas of 1 are assumed.

    Returns:
    -------
    samples : numpy.ndarray, shape (n,)
        An array of n samples drawn from the specified normal mixture.

    Raises:
    ------
    ValueError
        If inputs are invalid, such as weights not summing to 1, negative weights,
        mismatched lengths of weights, means, and sigmas, or non-positive sigmas.

    Examples:
    --------
    >>> import matplotlib.pyplot as plt
    >>> weights = [0.5, 0.3, 0.2]
    >>> means = [0, 5, 10]
    >>> sigmas = [1, 2, 3]
    >>> samples = normal_mix_variates(10000, weights, means, sigmas)
    >>> plt.hist(samples, bins=100, density=True, alpha=0.6, color='g')
    >>> plt.title('Histogram of Samples from Normal Mixture')
    >>> plt.show()
    """
    # Convert weights, means, and sigmas to NumPy arrays for vectorized operations
    weights = np.array(weights)

    # Input validation for weights
    if weights.ndim != 1:
        raise ValueError("Weights must be a one-dimensional array-like structure.")
    if np.any(weights < 0):
        raise ValueError("All mixture weights must be non-negative.")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-8):
        raise ValueError("Mixture weights must sum to 1.")

    k = len(weights)  # Number of components

    # Handle means
    if means is None:
        means = np.zeros(k)
    else:
        means = np.array(means)
        if means.shape != weights.shape:
            raise ValueError("Means must have the same length as weights.")

    # Handle sigmas
    if sigmas is None:
        sigmas = np.ones(k)
    else:
        sigmas = np.array(sigmas)
        if sigmas.shape != weights.shape:
            raise ValueError("Sigmas must have the same length as weights.")
        if np.any(sigmas <= 0):
            raise ValueError("All standard deviations (sigmas) must be positive.")

    # Input validation for n
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Number of samples 'n' must be a positive integer.")

    # Choose components for each sample based on the specified weights
    component_choices = np.random.choice(k, size=n, p=weights)

    # Generate samples from the chosen components
    # Preallocate the samples array
    samples = np.empty(n)

    # Iterate over each component to generate samples
    for i in range(k):
        # Find indices where component i was chosen
        indices = np.where(component_choices == i)[0]
        num_samples = len(indices)
        if num_samples > 0:
            # Draw samples for component i
            samples[indices] = np.random.normal(loc=means[i], scale=sigmas[i], size=num_samples)
    return samples

def fit_zero_mean_mixture(x, n_components=2, weights=None, sigmas=None, max_iter=100, tol=1e-6):
    """
    Fits a univariate normal mixture with zero means to data x using the EM algorithm.

    Args:
        x (np.ndarray): 1D array of data points.
        n_components (int): Number of mixture components.
        weights (np.ndarray): Initial weights for the components.
        sigmas (np.ndarray): Initial standard deviations for the components.
        max_iter (int): Maximum number of EM iterations.
        tol (float): Tolerance for convergence based on log-likelihood improvement.

    Returns:
        dict: A dictionary containing 'weights', 'sigmas', 'loglik', 'AIC', 'BIC'.
    """
    x = x.flatten()
    n_samples = x.shape[0]

    # Initialize parameters
    if weights is None:
        weights = np.full(n_components, 1 / n_components)
    else:
        weights = np.asarray(weights)
    if sigmas is None:
        sigmas = np.random.rand(n_components) + 0.5
    else:
        sigmas = np.asarray(sigmas)

    if len(weights) != n_components or len(sigmas) != n_components:
        raise ValueError("Length of weights and sigmas must be equal to n_components.")

    log_likelihood = -np.inf
    for iteration in range(max_iter):
        # E-step: Compute responsibilities
        log_resp = np.zeros((n_samples, n_components))
        for k in range(n_components):
            log_resp[:, k] = np.log(weights[k]) + norm.logpdf(x, loc=0, scale=sigmas[k])
        log_sum_resp = logsumexp(log_resp, axis=1)
        log_likelihood_new = np.sum(log_sum_resp)
        resp = np.exp(log_resp - log_sum_resp[:, np.newaxis])

        # Check convergence
        if np.abs(log_likelihood_new - log_likelihood) < tol:
            break
        log_likelihood = log_likelihood_new

        # M-step: Update weights and sigmas
        weights = resp.sum(axis=0) / n_samples
        for k in range(n_components):
            # Since mean is zero, we only update sigma
            sigmas[k] = np.sqrt(np.sum(resp[:, k] * (x ** 2)) / (resp[:, k].sum()))

    # Compute AIC and BIC
    n_params = n_components - 1 + n_components  # weights and sigmas
    AIC = 2 * n_params - 2 * log_likelihood
    BIC = n_params * np.log(n_samples) - 2 * log_likelihood

    return {
        'weights': weights,
        'sigmas': sigmas,
        'loglik': log_likelihood,
        'AIC': AIC,
        'BIC': BIC,
        '#iter': iteration
    }


def fit_mixture(x, n_components=2, weights=None, means=None, sigmas=None, max_iter=100, tol=1e-6):
    """
    Fits a univariate normal mixture with non-zero means to data x using the EM algorithm.

    Args:
        x (np.ndarray): 1D array of data points.
        n_components (int): Number of mixture components.
        weights (np.ndarray): Initial weights for the components.
        means (np.ndarray): Initial means for the components.
        sigmas (np.ndarray): Initial standard deviations for the components.
        max_iter (int): Maximum number of EM iterations.
        tol (float): Tolerance for convergence based on log-likelihood improvement.

    Returns:
        dict: A dictionary containing 'weights', 'means', 'sigmas', 'loglik', 'AIC', 'BIC'.
    """
    x = x.flatten()
    n_samples = x.shape[0]

    # Initialize parameters
    if weights is None:
        weights = np.full(n_components, 1 / n_components)
    else:
        weights = np.asarray(weights)
    if means is None:
        means = np.random.choice(x, n_components)
    else:
        means = np.asarray(means)
    if sigmas is None:
        sigmas = np.random.rand(n_components) + 0.5
    else:
        sigmas = np.asarray(sigmas)

    if len(weights) != n_components or len(means) != n_components or len(sigmas) != n_components:
        raise ValueError("Lengths of weights, means, and sigmas must be equal to n_components.")

    log_likelihood = -np.inf
    for iteration in range(max_iter):
        # E-step: Compute responsibilities
        log_resp = np.zeros((n_samples, n_components))
        for k in range(n_components):
            log_resp[:, k] = np.log(weights[k]) + norm.logpdf(x, loc=means[k], scale=sigmas[k])
        log_sum_resp = logsumexp(log_resp, axis=1)
        log_likelihood_new = np.sum(log_sum_resp)
        resp = np.exp(log_resp - log_sum_resp[:, np.newaxis])

        # Check convergence
        if np.abs(log_likelihood_new - log_likelihood) < tol:
            break
        log_likelihood = log_likelihood_new

        # M-step: Update weights, means, and sigmas
        weights = resp.sum(axis=0) / n_samples
        for k in range(n_components):
            resp_k = resp[:, k]
            total_resp_k = resp_k.sum()
            means[k] = np.sum(resp_k * x) / total_resp_k
            sigmas[k] = np.sqrt(np.sum(resp_k * ((x - means[k]) ** 2)) / total_resp_k)

    # Compute AIC and BIC
    n_params = n_components - 1 + 2 * n_components  # weights, means, sigmas
    AIC = 2 * n_params - 2 * log_likelihood
    BIC = n_params * np.log(n_samples) - 2 * log_likelihood

    return {
        'weights': weights,
        'means': means,
        'sigmas': sigmas,
        'loglik': log_likelihood,
        'AIC': AIC,
        'BIC': BIC,
        '#iter': iteration
    }

def simulate_zero_mean_mixture(n_samples=1000, weights=[0.5, 0.5], sigmas=[1.0, 2.0], random_state=None):
    """
    Simulates data from a zero-mean univariate normal mixture.

    Args:
        n_samples (int): Number of data points to simulate.
        weights (list or np.ndarray): Mixture weights (must sum to 1).
        sigmas (list or np.ndarray): Standard deviations of the components.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        np.ndarray: Simulated data.
    """
    if random_state is not None:
        np.random.seed(random_state)

    weights = np.array(weights)
    sigmas = np.array(sigmas)
    n_components = len(weights)
    # Ensure weights sum to 1
    weights /= weights.sum()
    # Choose component for each sample
    components = np.random.choice(n_components, size=n_samples, p=weights)
    x = np.random.normal(loc=0, scale=sigmas[components])
    return x

def hypsec_mix_stats(wgt, xmean=None, xsd=None):
    """
    Compute the mean, standard deviation, skewness, and excess kurtosis
    of a mixture of location-scale hyperbolic secant components.

    Each component is assumed to have:
        - mean = xmean[k]
        - variance = xsd[k]**2
        - third central moment = 0
        - fourth central moment = 5 * xsd[k]**4
      (standard hyperbolic secant has mean 0, var 1, and mu4 = 5).
    """
    n_components = len(wgt)
    if xmean is None:
        xmean = np.zeros(n_components)
    if xsd is None:
        xsd = np.ones(n_components)

    assert len(wgt) == len(xmean) == len(xsd), \
        "Arrays wgt, xmean, and xsd must have the same length"

    wgt = np.asarray(wgt, dtype=float)
    xmean = np.asarray(xmean, dtype=float)
    xsd = np.asarray(xsd, dtype=float)

    # normalize weights
    wgt = wgt / np.sum(wgt)

    # mixture mean
    mean = np.sum(wgt * xmean)

    # component offsets from mixture mean
    delta = xmean - mean
    var_comp = xsd**2

    # mixture variance
    variance = np.sum(wgt * (var_comp + delta**2))
    sd = np.sqrt(variance)

    # third central moment (component mu3 = 0)
    third_central_moment = np.sum(wgt * (delta**3 + 3.0 * delta * var_comp))
    skew = third_central_moment / sd**3

    # fourth central moment (component mu4 = 5 * xsd**4)
    mu4_comp = 5.0 * xsd**4
    fourth_central_moment = np.sum(
        wgt * (mu4_comp + 6.0 * delta**2 * var_comp + 3.0 * delta**4)
    )
    kurt_excess = fourth_central_moment / sd**4 - 3.0

    return {
        "mean": mean,
        "sd": sd,
        "skew": skew,
        "kurtosis": kurt_excess,
    }

def fit_two_normal_mixture_from_moments(w1, m1, m2, m3, m4, tol=1e-8, maxfev=10000):
    """
    Fit mu1, mu2, sd1, sd2 of a 2-normal mixture with weight w1 to match raw moments m1..m4.
    """

    # mixture weight must be strictly between 0 and 1
    if not (0.0 < w1 < 1.0):
        return np.nan, np.nan, np.nan, np.nan

    w2 = 1.0 - w1

    # system of equations: mixture raw moments minus targets
    def moment_equations(theta):
        mu1, mu2, a1, a2 = theta
        sd1 = np.exp(a1)  # enforce positivity
        sd2 = np.exp(a2)

        # component raw moments for normal N(mu, sd^2)
        # E[X]   = mu
        # E[X^2] = mu^2 + sd^2
        # E[X^3] = mu^3 + 3*mu*sd^2
        # E[X^4] = mu^4 + 6*mu^2*sd^2 + 3*sd^4

        # mixture moments
        m1_mix = w1 * mu1 + w2 * mu2

        m2_mix = w1 * (mu1**2 + sd1**2) + \
                 w2 * (mu2**2 + sd2**2)

        m3_mix = w1 * (mu1**3 + 3.0 * mu1 * sd1**2) + \
                 w2 * (mu2**3 + 3.0 * mu2 * sd2**2)

        m4_mix = w1 * (mu1**4 + 6.0 * mu1**2 * sd1**2 + 3.0 * sd1**4) + \
                 w2 * (mu2**4 + 6.0 * mu2**2 * sd2**2 + 3.0 * sd2**4)

        return np.array([
            m1_mix - m1,
            m2_mix - m2,
            m3_mix - m3,
            m4_mix - m4,
        ])

    # basic variance-based scale for initial guesses
    var = max(m2 - m1**2, 1e-8)
    base_sd = np.sqrt(var)

    # a small set of initial guesses to improve robustness
    x0_list = []

    # symmetric means around m1, equal sds
    mu1_0 = m1 - base_sd
    mu2_0 = m1 + base_sd
    sd1_0 = base_sd
    sd2_0 = base_sd
    x0_list.append(np.array([mu1_0, mu2_0, np.log(sd1_0), np.log(sd2_0)]))

    # both means at m1, unequal sds
    mu1_1 = m1
    mu2_1 = m1
    sd1_1 = base_sd * 0.5
    sd2_1 = base_sd * 2.0
    x0_list.append(np.array([mu1_1, mu2_1, np.log(sd1_1), np.log(sd2_1)]))

    # if nearly symmetric (m3 ~ 0), try a tighter symmetric guess
    if abs(m3) < 1e-6:
        mu1_2 = m1 - 0.5 * base_sd
        mu2_2 = m1 + 0.5 * base_sd
        sd_2 = base_sd
        x0_list.append(np.array([mu1_2, mu2_2, np.log(sd_2), np.log(sd_2)]))

    best_sol = None
    best_resid = np.inf

    for x0 in x0_list:
        try:
            sol = root(
                moment_equations,
                x0,
                method="hybr",
                tol=tol,
                options={"maxfev": maxfev},
            )
        except Exception:
            continue

        if not sol.success:
            continue

        # check residual
        res = moment_equations(sol.x)
        max_abs_resid = np.max(np.abs(res))
        if max_abs_resid < best_resid:
            best_resid = max_abs_resid
            best_sol = sol.x

    # if nothing converged well, declare failure
    if best_sol is None or best_resid > 1e-4:
        return np.nan, np.nan, np.nan, np.nan

    mu1, mu2, a1, a2 = best_sol
    sd1 = np.exp(a1)
    sd2 = np.exp(a2)

    # final sanity check: positive sds and finite numbers
    if not (np.isfinite(mu1) and np.isfinite(mu2) and
            np.isfinite(sd1) and np.isfinite(sd2) and
            sd1 > 0.0 and sd2 > 0.0):
        return np.nan, np.nan, np.nan, np.nan
    return mu1, mu2, sd1, sd2

def normal_mixture_moments_from_params(w1, mu1, mu2, sd1, sd2):
    """
    Compute raw moments m1..m4 for a 2-normal mixture given parameters.
    """
    w2 = 1.0 - w1

    # raw moments of N(mu, sd^2)
    def normal_moments(mu, sd):
        m1 = mu
        m2 = mu**2 + sd**2
        m3 = mu**3 + 3.0 * mu * sd**2
        m4 = mu**4 + 6.0 * mu**2 * sd**2 + 3.0 * sd**4
        return m1, m2, m3, m4

    m1_1, m2_1, m3_1, m4_1 = normal_moments(mu1, sd1)
    m1_2, m2_2, m3_2, m4_2 = normal_moments(mu2, sd2)

    m1 = w1 * m1_1 + w2 * m1_2
    m2 = w1 * m2_1 + w2 * m2_2
    m3 = w1 * m3_1 + w2 * m3_2
    m4 = w1 * m4_1 + w2 * m4_2

    return m1, m2, m3, m4


def sort_components_by_weight(w1, mu1, mu2, sd1, sd2):
    """
    Sort components in descending order of their weights.

    Parameters
    ----------
    w1 : float
        Weight of component 1 (component 2 has weight 1 - w1).
    mu1, mu2 : float
        Means of the two components.
    sd1, sd2 : float
        Standard deviations of the two components.

    Returns
    -------
    mu1_s, mu2_s, sd1_s, sd2_s : float
        Means and standard deviations reordered so that the first
        corresponds to the component with the larger weight.
    """
    w2 = 1.0 - w1
    comps = [(w1, mu1, sd1), (w2, mu2, sd2)]
    comps.sort(key=lambda t: t[0], reverse=True)
    _, mu1_s, sd1_s = comps[0]
    _, mu2_s, sd2_s = comps[1]
    return mu1_s, mu2_s, sd1_s, sd2_s

def fit_two_component_mixture_from_moments(
    w1, m1, m2, m3, m4, base_m2, base_m4, tol=1e-8, maxfev=10000
):
    """
    Fit mu1, mu2, sd1, sd2 of a 2-component mixture with weight w1 and
    symmetric zero-mean base distribution (E[Z^2]=base_m2, E[Z^4]=base_m4)
    to match raw moments m1..m4.
    """
    if not (0.0 < w1 < 1.0):
        return np.nan, np.nan, np.nan, np.nan
    if base_m2 <= 0.0 or base_m4 <= 0.0:
        return np.nan, np.nan, np.nan, np.nan

    w2 = 1.0 - w1

    def moment_equations(theta):
        mu1, mu2, a1, a2 = theta
        sd1 = np.exp(a1)  # enforce positivity
        sd2 = np.exp(a2)

        # base: Z has E[Z]=0, E[Z^2]=base_m2, E[Z^3]=0, E[Z^4]=base_m4
        # X = mu + sd*Z:
        # E[X]   = mu
        # E[X^2] = mu^2 + sd^2 * base_m2
        # E[X^3] = mu^3 + 3*mu*sd^2*base_m2
        # E[X^4] = mu^4 + 6*mu^2*sd^2*base_m2 + sd^4*base_m4

        m1_mix = w1 * mu1 + w2 * mu2

        m2_mix = (
            w1 * (mu1**2 + sd1**2 * base_m2)
            + w2 * (mu2**2 + sd2**2 * base_m2)
        )

        m3_mix = (
            w1 * (mu1**3 + 3.0 * mu1 * sd1**2 * base_m2)
            + w2 * (mu2**3 + 3.0 * mu2 * sd2**2 * base_m2)
        )

        m4_mix = (
            w1 * (mu1**4 + 6.0 * mu1**2 * sd1**2 * base_m2 + sd1**4 * base_m4)
            + w2 * (mu2**4 + 6.0 * mu2**2 * sd2**2 * base_m2 + sd2**4 * base_m4)
        )

        return np.array([
            m1_mix - m1,
            m2_mix - m2,
            m3_mix - m3,
            m4_mix - m4,
        ])

    # basic variance-based scale for initial guesses
    var_x = max(m2 - m1**2, 1e-8)
    base_sd = np.sqrt(var_x / base_m2)

    x0_list = []

    # symmetric means around m1, equal sds
    mu1_0 = m1 - base_sd
    mu2_0 = m1 + base_sd
    sd1_0 = base_sd
    sd2_0 = base_sd
    x0_list.append(np.array([mu1_0, mu2_0, np.log(sd1_0), np.log(sd2_0)]))

    # both means at m1, unequal sds
    mu1_1 = m1
    mu2_1 = m1
    sd1_1 = base_sd * 0.5
    sd2_1 = base_sd * 2.0
    x0_list.append(np.array([mu1_1, mu2_1, np.log(sd1_1), np.log(sd2_1)]))

    # if nearly symmetric (m3 ~ 0), try a tighter symmetric guess
    if abs(m3) < 1e-6:
        mu1_2 = m1 - 0.5 * base_sd
        mu2_2 = m1 + 0.5 * base_sd
        sd_2 = base_sd
        x0_list.append(np.array([mu1_2, mu2_2, np.log(sd_2), np.log(sd_2)]))

    best_sol = None
    best_resid = np.inf

    for x0 in x0_list:
        try:
            sol = root(
                moment_equations,
                x0,
                method="hybr",
                tol=tol,
                options={"maxfev": maxfev},
            )
        except Exception:
            continue

        if not sol.success:
            continue

        res = moment_equations(sol.x)
        max_abs_resid = np.max(np.abs(res))
        if max_abs_resid < best_resid:
            best_resid = max_abs_resid
            best_sol = sol.x

    if best_sol is None or best_resid > 1e-4:
        return np.nan, np.nan, np.nan, np.nan

    mu1, mu2, a1, a2 = best_sol
    sd1 = np.exp(a1)
    sd2 = np.exp(a2)

    if not (np.isfinite(mu1) and np.isfinite(mu2) and
            np.isfinite(sd1) and np.isfinite(sd2) and
            sd1 > 0.0 and sd2 > 0.0):
        return np.nan, np.nan, np.nan, np.nan

    return mu1, mu2, sd1, sd2

#!/usr/bin/env python3
"""
EM fit for a 1D K-component mixture of a common location-scale base density.

Model:
  p(x) = sum_{k=1..K} w[k] * (1/scale[k]) * f((x - loc[k]) / scale[k])

You supply f via base_pdf(z) (and optionally base_logpdf(z)).
Each component has its own (w, loc, scale); the base shape is shared.
"""

def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    return out.squeeze(axis=axis)

def fit_locscale_mixture_em(
    x,
    k,
    base_pdf,
    base_logpdf=None,
    *,
    max_iter=200,
    tol=1e-6,
    min_scale=1e-6,
    max_scale=None,
    n_init=3,
    seed=123,
    verbose=False,
):
    """
    Fit mixture parameters by maximum likelihood using EM.

    Parameters
    ----------
    x : array-like, shape (n,)
    k : int, number of mixture components
    base_pdf : callable(z) -> pdf values for standardized variable z
    base_logpdf : optional callable(z) -> logpdf values for standardized z
    max_iter : int, EM iterations per restart
    tol : float, stop when loglik improvement < tol
    min_scale : float, lower bound on component scale
    max_scale : float or None, upper bound on component scale (defaults to 100*std(x))
    n_init : int, number of random restarts (best loglik returned)
    seed : int, RNG seed
    verbose : bool

    Returns
    -------
    result : dict with keys: weights, loc, scale, loglik, responsibilities, n_iter
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n == 0:
        raise ValueError("x must be non-empty")
    if k < 1:
        raise ValueError("k must be >= 1")

    rng = np.random.default_rng(seed)

    x_std = float(np.std(x)) if n > 1 else 1.0
    if not np.isfinite(x_std) or x_std <= 0.0:
        x_std = 1.0
    if max_scale is None:
        max_scale = max(min_scale * 10.0, 100.0 * x_std)

    log_min_scale = math.log(min_scale)
    log_max_scale = math.log(max_scale)

    tiny = np.finfo(float).tiny

    def _base_logpdf(z):
        if base_logpdf is not None:
            out = base_logpdf(z)
            out = np.asarray(out, dtype=float)
            return out
        pdf = np.asarray(base_pdf(z), dtype=float)
        return np.log(np.maximum(pdf, tiny))

    def _log_component_pdf(xv, loc, scale):
        # log[(1/scale) * f((x-loc)/scale)] = -log(scale) + log f(z)
        z = (xv - loc) / scale
        return -math.log(scale) + _base_logpdf(z)

    def _e_step(xv, w, loc, scale):
        # log p_ik = log w_k + log comp_pdf_k(x_i)
        logw = np.log(np.maximum(w, tiny))
        logp = np.empty((n, k), dtype=float)
        for j in range(k):
            logp[:, j] = logw[j] + _log_component_pdf(xv, loc[j], scale[j])
        log_denom = _logsumexp(logp, axis=1)  # shape (n,)
        r = np.exp(logp - log_denom[:, None])
        ll = float(np.sum(log_denom))
        return r, ll

    def _m_step_weights(r):
        wk = r.sum(axis=0)
        w = wk / max(wk.sum(), tiny)
        return w, wk

    def _m_step_locscale_one_component(xv, rj, loc0, scale0):
        # maximize sum_i r_i * (-log s + log f((x_i-loc)/s))
        # optimize over (loc, log_scale) with bounds
        rj = np.asarray(rj, dtype=float)
        rsum = float(rj.sum())
        if not np.isfinite(rsum) or rsum <= 0.0:
            return loc0, max(scale0, min_scale), False

        def nll(params):
            loc = float(params[0])
            log_scale = float(params[1])
            # clip log_scale inside bounds to avoid invalid exp in line search
            log_scale = min(max(log_scale, log_min_scale), log_max_scale)
            scale = math.exp(log_scale)
            z = (xv - loc) / scale
            ll_i = -log_scale + _base_logpdf(z)
            # negative weighted log-likelihood
            return -float(np.sum(rj * ll_i))

        x0 = np.array([loc0, math.log(max(scale0, min_scale))], dtype=float)
        bnds = [(None, None), (log_min_scale, log_max_scale)]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bnds)

        if (not res.success) or (not np.all(np.isfinite(res.x))):
            return loc0, max(scale0, min_scale), False

        loc_hat = float(res.x[0])
        log_scale_hat = float(res.x[1])
        log_scale_hat = min(max(log_scale_hat, log_min_scale), log_max_scale)
        scale_hat = math.exp(log_scale_hat)
        scale_hat = max(scale_hat, min_scale)
        return loc_hat, scale_hat, True

    def _init_params(xv):
        # deterministic-ish init: pick locs near quantiles + small noise; scales ~ std
        qs = (np.arange(k) + 0.5) / k
        loc = np.quantile(xv, qs)
        loc = loc + rng.normal(scale=0.05 * x_std, size=k)
        scale = np.full(k, max(x_std, min_scale), dtype=float)
        w = np.full(k, 1.0 / k, dtype=float)
        return w, loc.astype(float), scale

    best = None

    for init_id in range(n_init):
        w, loc, scale = _init_params(x)

        prev_ll = -np.inf
        r = None
        n_iter_done = 0

        for it in range(1, max_iter + 1):
            r, ll = _e_step(x, w, loc, scale)

            if verbose:
                print(f"init {init_id+1}/{n_init} iter {it:4d} ll {ll: .6f}")

            # convergence
            if np.isfinite(prev_ll) and (ll - prev_ll) < tol:
                n_iter_done = it
                break
            prev_ll = ll

            # M-step: weights
            w, wk = _m_step_weights(r)

            # handle empty components by mild reinit
            empty = wk <= 1e-12
            if np.any(empty):
                for j in np.where(empty)[0]:
                    loc[j] = float(rng.choice(x))
                    scale[j] = max(x_std, min_scale)
                    w[j] = 1.0 / k
                w = w / w.sum()

            # M-step: (loc, scale) per component via weighted numerical MLE
            for j in range(k):
                loc_j, scale_j, ok = _m_step_locscale_one_component(
                    x, r[:, j], loc[j], scale[j]
                )
                loc[j] = loc_j
                scale[j] = scale_j

            n_iter_done = it

        # final E-step to get final ll and responsibilities
        r, ll = _e_step(x, w, loc, scale)

        cand = {
            "weights": w.copy(),
            "loc": loc.copy(),
            "scale": scale.copy(),
            "loglik": float(ll),
            "responsibilities": r,
            "n_iter": int(n_iter_done),
        }
        if (best is None) or (cand["loglik"] > best["loglik"]):
            best = cand

    return best

def simulate_t_mixture(n, weights, loc, scale, df, seed=123):
    """ Simulate n draws from a K-component 1D mixture of Student t
    distributions with common df and component-specific weights, loc, and
    scale. """
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    k = len(weights)
    z = rng.choice(k, size=n, p=weights)

    x = np.empty(n, dtype=float)
    for j in range(k):
        idx = (z == j)
        nj = int(idx.sum())
        if nj > 0:
            x[idx] = student_t.rvs(df=df, loc=loc[j], scale=scale[j], size=nj, random_state=rng)
    return x

def simulate_2_normal_mix(n, p1, m1, m2, sd1, sd2):
    """Simulate data from a mixture of two normal distributions."""
    # Simulate component labels
    labels = np.random.choice([0, 1], size=n, p=[p1, 1-p1])
    # Simulate data from the normal distributions
    x = np.where(labels == 0,
                 np.random.normal(m1, sd1, size=n),
                 np.random.normal(m2, sd2, size=n))
    return x

def fit_2norm_mix_zero_mean_m2_1(m4, m6=None, mean_abs=None, tol=1e-12, max_iter=200):
    """
    Return (w1, sigma1, sigma2) for a 2-component zero-mean normal mixture
    matching m2=1 and m4, plus either m6 or E|X|.
    Components are ordered so sigma1 <= sigma2 and w1 is the weight on sigma1.
    """
    m4 = float(m4)
    if m4 <= 3.0:
        raise ValueError("need m4 > 3 for a genuine 2-component solution")

    if (m6 is None) == (mean_abs is None):
        raise ValueError("provide exactly one of m6 or mean_abs")

    def _order(w, s1, s2):
        if s1 <= s2:
            return float(w), float(s1), float(s2)
        return float(1.0 - w), float(s2), float(s1)

    if m6 is not None:
        m6 = float(m6)

        # necessary condition (also essentially required for feasibility)
        if m6 < m4 * m4:
            raise ValueError("infeasible moments for any distribution: need m6 >= m4^2")

        k2 = m4 / 3.0
        k3 = m6 / 15.0

        s2 = k2 - 1.0
        if s2 <= 0.0:
            raise ValueError("infeasible: k2 must be > 1")

        # third central moment of V where V takes values v1,v2 with weights w,1-w
        s3 = k3 - 3.0 * k2 + 2.0

        r = s3 / (s2 ** 1.5)
        t = r / np.sqrt(r * r + 4.0)

        w = 0.5 * (1.0 + t)
        q = 1.0 - w

        if not (0.0 < w < 1.0):
            raise ValueError("degenerate solution (weight not in (0,1))")

        d = np.sqrt(s2 / (w * q))

        v1 = 1.0 - q * d
        v2 = 1.0 + w * d

        if v1 <= 0.0 or v2 <= 0.0:
            raise ValueError("no valid 2-normal mixture: implied variance is nonpositive")

        sigma1 = np.sqrt(v1)
        sigma2 = np.sqrt(v2)

        w1, sigma1, sigma2 = _order(w, sigma1, sigma2)
        return {"w1": w1, "w2": 1.0 - w1, "sigma1": sigma1, "sigma2": sigma2}

    # else: mean_abs is given
    a = float(mean_abs)
    k = m4 / 3.0  # > 1

    a_max = np.sqrt(2.0 / np.pi)             # limit as p -> 0 (approaches N(0,1))
    a_min = np.sqrt(2.0 / np.pi) / np.sqrt(k)  # limit as p -> 1/k (low-var -> 0)

    if not (a_min < a < a_max):
        raise ValueError(f"infeasible mean_abs for 2-normal mixture: need {a_min} < mean_abs < {a_max}")

    def a_of_p(p):
        # p is the weight on the high-variance component; 1-p on the low-variance component
        p = float(p)
        q = 1.0 - p
        v_hi = 1.0 + np.sqrt((k - 1.0) * q / p)
        v_lo = 1.0 - np.sqrt((k - 1.0) * p / q)
        if v_lo <= 0.0:
            return -np.inf
        return np.sqrt(2.0 / np.pi) * (q * np.sqrt(v_lo) + p * np.sqrt(v_hi))

    # p lives in (0, 1/k); a_of_p is decreasing on that interval
    p_lo = 1e-16
    p_hi = (1.0 / k) - 1e-16

    # bisection for a_of_p(p) = a
    f_lo = a_of_p(p_lo) - a
    f_hi = a_of_p(p_hi) - a
    if not (f_lo > 0.0 and f_hi < 0.0):
        raise ValueError("bisection bracket failed (numerical issue)")

    for _ in range(max_iter):
        p_mid = 0.5 * (p_lo + p_hi)
        f_mid = a_of_p(p_mid) - a
        if abs(f_mid) <= tol:
            p_lo = p_hi = p_mid
            break
        if f_mid > 0.0:
            p_lo = p_mid
        else:
            p_hi = p_mid

    p = 0.5 * (p_lo + p_hi)
    q = 1.0 - p
    v_hi = 1.0 + np.sqrt((k - 1.0) * q / p)
    v_lo = 1.0 - np.sqrt((k - 1.0) * p / q)

    sigma_hi = np.sqrt(v_hi)
    sigma_lo = np.sqrt(v_lo)

    # order so sigma1 <= sigma2 and w1 is weight on sigma1
    w1, sigma1, sigma2 = _order(q, sigma_lo, sigma_hi)  # weight on low-var comp is q
    return {"w1": w1, "w2": 1.0 - w1, "sigma1": sigma1, "sigma2": sigma2}
