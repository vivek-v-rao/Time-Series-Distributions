""" fit GED and SGED distributions to i.i.d. data """
import numpy as np
from scipy import stats
from scipy.optimize import minimize

from ar_sged_model import implied_sged_moments, sged_logpdf

LARGE_PENALTY = 1.0e100


def fit_normal(x):
    """Fit a zero-location normal distribution to data by maximum likelihood."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("no finite observations")
    scale = np.sqrt(np.mean(x * x))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    llf = stats.norm.logpdf(x, loc=0.0, scale=scale).sum()
    npar = 1
    nobs = len(x)
    return {
        "scale": float(scale),
        "llf": float(llf),
        "aic": float(2 * npar - 2 * llf),
        "bic": float(np.log(nobs) * npar - 2 * llf),
        "implied_mean": 0.0,
        "implied_var": float(scale * scale),
        "implied_skew": 0.0,
        "implied_excess_kurtosis": 0.0,
        "nobs_fit": nobs,
    }


def fit_ged(x):
    """Fit a zero-location GED to data by maximum likelihood."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("no finite observations")
    scale0 = np.std(x)
    if not np.isfinite(scale0) or scale0 <= 0.0:
        scale0 = 1.0
    theta0 = np.array([np.log(scale0), np.log(2.0)], dtype=float)

    def neg_ll(theta):
        scale = np.exp(theta[0])
        beta = np.exp(theta[1])
        if not np.isfinite(scale) or not np.isfinite(beta):
            return LARGE_PENALTY
        if scale <= 0.0 or beta <= 0.0:
            return LARGE_PENALTY
        ll = stats.gennorm.logpdf(x, beta=beta, loc=0.0, scale=scale).sum()
        return -ll if np.isfinite(ll) else LARGE_PENALTY

    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("optimization failed: " + res.message)
    scale = np.exp(res.x[0])
    beta = np.exp(res.x[1])
    llf = -res.fun
    npar = 2
    nobs = len(x)
    mean0, var0, skew0, ex_kurt0 = stats.gennorm.stats(
        beta=beta, loc=0.0, scale=scale, moments="mvsk"
    )
    return {
        "scale": scale,
        "beta": beta,
        "llf": llf,
        "aic": 2 * npar - 2 * llf,
        "bic": np.log(nobs) * npar - 2 * llf,
        "nobs_fit": nobs,
        "implied_mean": float(mean0),
        "implied_var": float(var0),
        "implied_skew": float(skew0),
        "implied_excess_kurtosis": float(ex_kurt0),
    }


def fit_sged(x):
    """Fit a zero-location SGED to data by maximum likelihood."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("no finite observations")
    scale0 = np.std(x)
    if not np.isfinite(scale0) or scale0 <= 0.0:
        scale0 = 1.0
    theta0 = np.array([np.log(scale0), np.log(2.0), 0.0], dtype=float)

    def neg_ll(theta):
        scale = np.exp(theta[0])
        beta = np.exp(theta[1])
        xi = np.exp(theta[2])
        if not np.isfinite(scale) or not np.isfinite(beta) or not np.isfinite(xi):
            return LARGE_PENALTY
        if scale <= 0.0 or beta <= 0.0 or xi <= 0.0:
            return LARGE_PENALTY
        ll = sged_logpdf(x, beta=beta, xi=xi, scale=scale).sum()
        return -ll if np.isfinite(ll) else LARGE_PENALTY

    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("optimization failed: " + res.message)
    scale = np.exp(res.x[0])
    beta = np.exp(res.x[1])
    xi = np.exp(res.x[2])
    llf = -res.fun
    npar = 3
    nobs = len(x)
    moments = implied_sged_moments(beta=beta, xi=xi, scale=scale)
    return {
        "scale": scale,
        "beta": beta,
        "xi": xi,
        "llf": llf,
        "aic": 2 * npar - 2 * llf,
        "bic": np.log(nobs) * npar - 2 * llf,
        "nobs_fit": nobs,
        "implied_mean": moments["mean"],
        "implied_var": moments["var"],
        "implied_skew": moments["skew"],
        "implied_excess_kurtosis": moments["excess_kurtosis"],
    }


def print_fit_summary(label, fit):
    """Print a short summary of a GED or SGED fit."""
    print(f"\n{label}")
    if "xi" in fit:
        print("beta:", f"{fit['beta']:.3f}", "xi:", f"{fit['xi']:.3f}",
            "scale:", f"{fit['scale']:.3f}", "AIC:", f"{fit['aic']:.3f}",
            "BIC:", f"{fit['bic']:.3f}")
        print("implied mean:", f"{fit['implied_mean']:.3f}",
            "implied skew:", f"{fit['implied_skew']:.3f}",
            "implied ex kurt:", f"{fit['implied_excess_kurtosis']:.3f}")
    else:
        print("beta:", f"{fit['beta']:.3f}", "scale:", f"{fit['scale']:.3f}",
            "AIC:", f"{fit['aic']:.3f}", "BIC:", f"{fit['bic']:.3f}")
        print("implied mean:", f"{fit['implied_mean']:.3f}",
            "implied skew:", f"{fit['implied_skew']:.3f}",
            "implied ex kurt:", f"{fit['implied_excess_kurtosis']:.3f}")
