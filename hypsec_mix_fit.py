""" fit mixtures of hyperbolic secant distributions to i.i.d. data """
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import hypsecant

from mix import hypsec_mix_stats

LARGE_PENALTY = 1.0e100


def _softmax_last_zero(logits):
    """Map unconstrained logits to simplex weights."""
    logits = np.asarray(logits, dtype=float)
    logits_full = np.append(logits, 0.0)
    logits_full = logits_full - np.max(logits_full)
    weights = np.exp(logits_full)
    return weights / np.sum(weights)


def fit_hypsec_mix(x, n_components=2):
    """Fit an n-component hyperbolic secant mixture by MLE."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("no finite observations")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    mean0 = float(np.mean(x))
    sd0 = float(np.std(x))
    if not np.isfinite(sd0) or sd0 <= 0.0:
        sd0 = 1.0
    if n_components == 1:
        mean_init = np.array([mean0], dtype=float)
        sd_init = np.array([sd0], dtype=float)
        theta0 = np.concatenate([mean_init, np.log(sd_init)])
    else:
        mean_init = np.linspace(mean0 - 0.5 * sd0, mean0 + 0.5 * sd0,
            n_components)
        sd_init = np.linspace(0.75 * sd0, 1.25 * sd0, n_components)
        theta0 = np.concatenate([
            np.zeros(n_components - 1, dtype=float),
            mean_init,
            np.log(sd_init),
        ])

    def neg_ll(theta):
        if n_components == 1:
            weights = np.array([1.0], dtype=float)
            means = np.array([theta[0]], dtype=float)
            sds = np.exp(np.array([theta[1]], dtype=float))
        else:
            weights = _softmax_last_zero(theta[:n_components - 1])
            means = np.array(theta[n_components - 1:2 * n_components - 1],
                dtype=float)
            sds = np.exp(np.array(theta[2 * n_components - 1:], dtype=float))
        if np.any(~np.isfinite(weights)) or np.any(~np.isfinite(means)):
            return LARGE_PENALTY
        if np.any(~np.isfinite(sds)) or np.any(sds <= 0.0):
            return LARGE_PENALTY
        logpdf = np.column_stack([
            np.log(weights[icomp]) + hypsecant.logpdf(
                x, loc=means[icomp], scale=sds[icomp]
            )
            for icomp in range(n_components)
        ])
        ll = np.sum(logsumexp(logpdf, axis=1))
        return -ll if np.isfinite(ll) else LARGE_PENALTY

    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("optimization failed: " + res.message)
    if n_components == 1:
        weights = np.array([1.0], dtype=float)
        means = np.array([res.x[0]], dtype=float)
        sds = np.exp(np.array([res.x[1]], dtype=float))
    else:
        weights = _softmax_last_zero(res.x[:n_components - 1])
        means = np.array(res.x[n_components - 1:2 * n_components - 1],
            dtype=float)
        sds = np.exp(np.array(res.x[2 * n_components - 1:], dtype=float))
    order = np.argsort(weights)[::-1]
    weights = weights[order]
    means = means[order]
    sds = sds[order]
    llf = float(-res.fun)
    npar = (n_components - 1) + n_components + n_components
    nobs = len(x)
    moms = hypsec_mix_stats(weights, means, sds)
    return {
        "n_components": int(n_components),
        "weights": weights,
        "means": means,
        "sds": sds,
        "llf": llf,
        "aic": float(2 * npar - 2 * llf),
        "bic": float(np.log(nobs) * npar - 2 * llf),
        "nobs_fit": nobs,
        "implied_mean": float(moms["mean"]),
        "implied_var": float(moms["sd"] ** 2),
        "implied_skew": float(moms["skew"]),
        "implied_excess_kurtosis": float(moms["kurtosis"]),
    }


def print_fit_summary(label, fit):
    """Print a short summary of a hyperbolic secant mixture fit."""
    print(f"\n{label}")
    print("n_components:", fit["n_components"], "AIC:", f"{fit['aic']:.3f}",
        "BIC:", f"{fit['bic']:.3f}")
    print("implied mean:", f"{fit['implied_mean']:.3f}",
        "implied skew:", f"{fit['implied_skew']:.3f}",
        "implied ex kurt:", f"{fit['implied_excess_kurtosis']:.3f}")
    print(f"{'comp':>8}{'weight':>10}{'mean':>10}{'sd':>10}")
    for icomp, (weight, mean, sd) in enumerate(
            zip(fit["weights"], fit["means"], fit["sds"]), start=1):
        print(f"{icomp:8d}{weight:10.3f}{mean:10.3f}{sd:10.3f}")
