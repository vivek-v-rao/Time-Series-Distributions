""" fit mixtures of normals to i.i.d. data """
import numpy as np
from sklearn.mixture import GaussianMixture


def mixture_moments(weights, means, variances):
    """Return mean, variance, skewness, and excess kurtosis of a normal mixture."""
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    mean = np.sum(weights * means)
    second = np.sum(weights * (variances + means * means))
    var = second - mean * mean
    third = np.sum(weights * ((means - mean) ** 3 + 3.0 * (means - mean) * variances))
    fourth = np.sum(weights * (
        (means - mean) ** 4
        + 6.0 * ((means - mean) ** 2) * variances
        + 3.0 * variances * variances
    ))
    skew = third / (var ** 1.5)
    ex_kurt = fourth / (var * var) - 3.0
    return {
        "mean": float(mean),
        "var": float(var),
        "skew": float(skew),
        "excess_kurtosis": float(ex_kurt),
    }


def fit_normal_mix(x, n_components=2, seed=0):
    """Fit a Gaussian mixture to data by MLE."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("no finite observations")
    gmm = GaussianMixture(n_components=n_components, random_state=seed)
    gmm.fit(x.reshape(-1, 1))
    weights = gmm.weights_.copy()
    means = gmm.means_.flatten().copy()
    variances = gmm.covariances_.reshape(-1).copy()
    order = np.argsort(weights)[::-1]
    weights = weights[order]
    means = means[order]
    variances = variances[order]
    llf = float(gmm.score(x.reshape(-1, 1)) * len(x))
    npar = (n_components - 1) + n_components + n_components
    moms = mixture_moments(weights, means, variances)
    return {
        "n_components": int(n_components),
        "weights": weights,
        "means": means,
        "variances": variances,
        "llf": llf,
        "aic": float(2 * npar - 2 * llf),
        "bic": float(np.log(len(x)) * npar - 2 * llf),
        "nobs_fit": len(x),
        "implied_mean": moms["mean"],
        "implied_var": moms["var"],
        "implied_skew": moms["skew"],
        "implied_excess_kurtosis": moms["excess_kurtosis"],
    }


def print_fit_summary(label, fit):
    """Print a short summary of a normal mixture fit."""
    print(f"\n{label}")
    print("n_components:", fit["n_components"], "AIC:", f"{fit['aic']:.3f}",
        "BIC:", f"{fit['bic']:.3f}")
    print("implied mean:", f"{fit['implied_mean']:.3f}",
        "implied skew:", f"{fit['implied_skew']:.3f}",
        "implied ex kurt:", f"{fit['implied_excess_kurtosis']:.3f}")
    print(f"{'comp':>8}{'weight':>10}{'mean':>10}{'sd':>10}")
    for icomp, (weight, mean, var) in enumerate(
            zip(fit["weights"], fit["means"], fit["variances"]), start=1):
        print(f"{icomp:8d}{weight:10.3f}{mean:10.3f}{np.sqrt(var):10.3f}")
