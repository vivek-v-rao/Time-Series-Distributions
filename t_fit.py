""" fit symmetric and skewed t distributions to i.i.d. data """
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma

LARGE_PENALTY = 1.0e100


def skewt_logpdf(x, df, xi, scale):
    """Fernandez-Steel style skewed t log-density with location 0."""
    z = np.asarray(x, dtype=float) / scale
    log_norm = np.log(2.0) - np.log(scale) - np.log(xi + 1.0 / xi)
    z_adj = np.where(z < 0.0, xi * z, z / xi)
    return log_norm + stats.t.logpdf(z_adj, df=df, loc=0.0, scale=1.0)


def implied_skewt_moments(df, xi, scale):
    """Return mean, variance, skewness, and excess kurtosis for the skewed t."""
    def abs_t_moment(order):
        if order >= df:
            raise ValueError("absolute moment does not exist for order >= df")
        return (
            (df ** (0.5 * order))
            * gamma(0.5 * (order + 1.0))
            * gamma(0.5 * (df - order))
            / (np.sqrt(np.pi) * gamma(0.5 * df))
        )

    def raw_moment(order):
        coeff = (
            xi ** (order + 1.0) + ((-1.0) ** order) * xi ** (-(order + 1.0))
        ) / (xi + 1.0 / xi)
        return (scale ** order) * coeff * abs_t_moment(order)

    m1 = raw_moment(1)
    m2 = raw_moment(2)
    m3 = raw_moment(3)
    m4 = raw_moment(4)
    var = m2 - m1 * m1
    mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    mu4 = m4 - 4.0 * m1 * m3 + 6.0 * (m1 ** 2) * m2 - 3.0 * m1 ** 4
    skew = mu3 / (var ** 1.5)
    ex_kurt = mu4 / (var ** 2) - 3.0
    return {
        "mean": float(m1),
        "var": float(var),
        "skew": float(skew),
        "excess_kurtosis": float(ex_kurt),
    }


def fit_t(x):
    """Fit a zero-location symmetric t distribution with df > 4."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("no finite observations")
    scale0 = np.std(x)
    if not np.isfinite(scale0) or scale0 <= 0.0:
        scale0 = 1.0
    theta0 = np.array([np.log(6.0 - 4.0), np.log(scale0)], dtype=float)

    def neg_ll(theta):
        df = 4.0 + np.exp(theta[0])
        scale = np.exp(theta[1])
        if not np.isfinite(df) or not np.isfinite(scale):
            return LARGE_PENALTY
        if df <= 4.0 or scale <= 0.0:
            return LARGE_PENALTY
        ll = stats.t.logpdf(x, df=df, loc=0.0, scale=scale).sum()
        return -ll if np.isfinite(ll) else LARGE_PENALTY

    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("optimization failed: " + res.message)
    df = float(4.0 + np.exp(res.x[0]))
    scale = float(np.exp(res.x[1]))
    llf = float(-res.fun)
    npar = 2
    nobs = len(x)
    var = scale * scale * df / (df - 2.0)
    ex_kurt = 6.0 / (df - 4.0)
    return {
        "df": df,
        "scale": scale,
        "llf": llf,
        "aic": float(2 * npar - 2 * llf),
        "bic": float(np.log(nobs) * npar - 2 * llf),
        "nobs_fit": nobs,
        "implied_mean": 0.0,
        "implied_var": float(var),
        "implied_skew": 0.0,
        "implied_excess_kurtosis": float(ex_kurt),
    }


def fit_skewt(x):
    """Fit a zero-location skewed t distribution with df > 4."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise ValueError("no finite observations")
    scale0 = np.std(x)
    if not np.isfinite(scale0) or scale0 <= 0.0:
        scale0 = 1.0
    theta0 = np.array([np.log(6.0 - 4.0), 0.0, np.log(scale0)], dtype=float)

    def neg_ll(theta):
        df = 4.0 + np.exp(theta[0])
        xi = np.exp(theta[1])
        scale = np.exp(theta[2])
        if not np.isfinite(df) or not np.isfinite(xi) or not np.isfinite(scale):
            return LARGE_PENALTY
        if df <= 4.0 or xi <= 0.0 or scale <= 0.0:
            return LARGE_PENALTY
        ll = skewt_logpdf(x, df=df, xi=xi, scale=scale).sum()
        return -ll if np.isfinite(ll) else LARGE_PENALTY

    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("optimization failed: " + res.message)
    df = float(4.0 + np.exp(res.x[0]))
    xi = float(np.exp(res.x[1]))
    scale = float(np.exp(res.x[2]))
    llf = float(-res.fun)
    npar = 3
    nobs = len(x)
    moments = implied_skewt_moments(df=df, xi=xi, scale=scale)
    return {
        "df": df,
        "xi": xi,
        "scale": scale,
        "llf": llf,
        "aic": float(2 * npar - 2 * llf),
        "bic": float(np.log(nobs) * npar - 2 * llf),
        "nobs_fit": nobs,
        "implied_mean": moments["mean"],
        "implied_var": moments["var"],
        "implied_skew": moments["skew"],
        "implied_excess_kurtosis": moments["excess_kurtosis"],
    }


def print_fit_summary(label, fit):
    """Print a short summary of a t or skewed t fit."""
    print(f"\n{label}")
    if "xi" in fit:
        print("df:", f"{fit['df']:.3f}", "xi:", f"{fit['xi']:.3f}",
            "scale:", f"{fit['scale']:.3f}", "AIC:", f"{fit['aic']:.3f}",
            "BIC:", f"{fit['bic']:.3f}")
    else:
        print("df:", f"{fit['df']:.3f}", "scale:", f"{fit['scale']:.3f}",
            "AIC:", f"{fit['aic']:.3f}", "BIC:", f"{fit['bic']:.3f}")
    print("implied mean:", f"{fit['implied_mean']:.3f}",
        "implied skew:", f"{fit['implied_skew']:.3f}",
        "implied ex kurt:", f"{fit['implied_excess_kurtosis']:.3f}")
