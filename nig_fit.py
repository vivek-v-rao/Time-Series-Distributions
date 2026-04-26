""" fit normal inverse Gaussian distributions to i.i.d. data """
import numpy as np
from scipy import stats
from scipy.optimize import minimize

LARGE_PENALTY = 1.0e100


def fit_nig(x):
    """Fit a zero-location NIG distribution to data by maximum likelihood."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("no finite observations")
    scale0 = np.std(x)
    if not np.isfinite(scale0) or scale0 <= 0.0:
        scale0 = 1.0
    theta0 = np.array([np.log(1.5), np.arctanh(0.0), np.log(scale0)], dtype=float)

    def neg_ll(theta):
        a = np.exp(theta[0]) + 1.0e-6
        rho = np.tanh(theta[1])
        b = a * rho
        scale = np.exp(theta[2])
        if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(scale):
            return LARGE_PENALTY
        if scale <= 0.0 or abs(b) >= a:
            return LARGE_PENALTY
        ll = stats.norminvgauss.logpdf(x, a=a, b=b, loc=0.0, scale=scale).sum()
        return -ll if np.isfinite(ll) else LARGE_PENALTY

    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("optimization failed: " + res.message)
    a = np.exp(res.x[0]) + 1.0e-6
    rho = np.tanh(res.x[1])
    b = a * rho
    scale = np.exp(res.x[2])
    llf = -res.fun
    npar = 3
    nobs = len(x)
    mean0, var0, skew0, ex_kurt0 = stats.norminvgauss.stats(
        a=a, b=b, loc=0.0, scale=scale, moments="mvsk"
    )
    return {
        "a": float(a),
        "b": float(b),
        "scale": float(scale),
        "llf": float(llf),
        "aic": float(2 * npar - 2 * llf),
        "bic": float(np.log(nobs) * npar - 2 * llf),
        "nobs_fit": nobs,
        "implied_mean": float(mean0),
        "implied_var": float(var0),
        "implied_skew": float(skew0),
        "implied_excess_kurtosis": float(ex_kurt0),
    }


def print_fit_summary(label, fit):
    """Print a short summary of an NIG fit."""
    print(f"\n{label}")
    print("a:", f"{fit['a']:.3f}", "b:", f"{fit['b']:.3f}",
        "scale:", f"{fit['scale']:.3f}", "AIC:", f"{fit['aic']:.3f}",
        "BIC:", f"{fit['bic']:.3f}")
    print("implied mean:", f"{fit['implied_mean']:.3f}",
        "implied skew:", f"{fit['implied_skew']:.3f}",
        "implied ex kurt:", f"{fit['implied_excess_kurtosis']:.3f}")
