""" fit generalized hyperbolic distributions to i.i.d. data """
import numpy as np
from scipy import stats
from scipy.optimize import minimize

LARGE_PENALTY = 1.0e100


def fit_gh(x, constrained=True, mgf1_margin=0.05):
    """Fit a zero-location generalized hyperbolic distribution by MLE."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("no finite observations")
    scale0 = np.std(x)
    if not np.isfinite(scale0) or scale0 <= 0.0:
        scale0 = 1.0
    theta0 = np.array([0.0, np.log(1.5), np.arctanh(0.0), np.log(scale0)],
        dtype=float)

    def neg_ll(theta):
        p = theta[0]
        a = np.exp(theta[1]) + 1.0e-6
        rho = np.tanh(theta[2])
        b = a * rho
        scale = np.exp(theta[3])
        if not np.isfinite(p) or not np.isfinite(a) or not np.isfinite(b):
            return LARGE_PENALTY
        if not np.isfinite(scale) or scale <= 0.0 or abs(b) >= a:
            return LARGE_PENALTY
        if constrained and a <= abs(b + 1.0) + mgf1_margin:
            return LARGE_PENALTY
        ll = stats.genhyperbolic.logpdf(
            x, p=p, a=a, b=b, loc=0.0, scale=scale
        ).sum()
        return -ll if np.isfinite(ll) else LARGE_PENALTY

    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("optimization failed: " + res.message)
    p = float(res.x[0])
    a = float(np.exp(res.x[1]) + 1.0e-6)
    rho = float(np.tanh(res.x[2]))
    b = float(a * rho)
    scale = float(np.exp(res.x[3]))
    llf = float(-res.fun)
    npar = 4
    nobs = len(x)
    mean0, var0, skew0, ex_kurt0 = stats.genhyperbolic.stats(
        p=p, a=a, b=b, loc=0.0, scale=scale, moments="mvsk"
    )
    return {
        "p": p,
        "a": a,
        "b": b,
        "scale": scale,
        "llf": llf,
        "aic": float(2 * npar - 2 * llf),
        "bic": float(np.log(nobs) * npar - 2 * llf),
        "nobs_fit": nobs,
        "implied_mean": float(mean0),
        "implied_var": float(var0),
        "implied_skew": float(skew0),
        "implied_excess_kurtosis": float(ex_kurt0),
        "constrained": bool(constrained),
        "mgf1_margin": float(mgf1_margin),
    }


def print_fit_summary(label, fit):
    """Print a short summary of a GH fit."""
    print(f"\n{label}")
    print("constrained:", fit["constrained"], "mgf1_margin:",
        f"{fit['mgf1_margin']:.3f}")
    print("p:", f"{fit['p']:.3f}", "a:", f"{fit['a']:.3f}",
        "b:", f"{fit['b']:.3f}", "scale:", f"{fit['scale']:.3f}",
        "AIC:", f"{fit['aic']:.3f}", "BIC:", f"{fit['bic']:.3f}")
    print("implied mean:", f"{fit['implied_mean']:.3f}",
        "implied skew:", f"{fit['implied_skew']:.3f}",
        "implied ex kurt:", f"{fit['implied_excess_kurtosis']:.3f}")
