""" fit autoregressive models with GED innovations """
import numpy as np
from scipy import stats
from scipy.optimize import minimize

def ar_design(ts, nar, trend):
    """Build response and design matrix for an AR model with optional trend terms."""
    nobs = len(ts)
    y = ts[nar:]
    xcols = []
    if "c" in trend:
        xcols.append(np.ones(len(y)))
    if "t" in trend:
        xcols.append(np.arange(nar, nobs, dtype=float))
    for ilag in range(1, nar + 1):
        xcols.append(ts[nar - ilag:nobs - ilag])
    x = np.column_stack(xcols) if xcols else np.empty((len(y), 0))
    prev_level = ts[nar - 1:nobs - 1]
    return y, x, prev_level

def fit_ar_ged(ts, nar=1, trend="c"):
    """Fit an AR model by conditional MLE under GED innovations."""
    if nar < 1:
        raise ValueError("nar must be positive")
    y, x, prev_level = ar_design(ts, nar, trend)
    if len(y) <= x.shape[1]:
        raise ValueError("not enough observations to estimate model")
    if x.shape[1] > 0:
        coeff0 = np.linalg.lstsq(x, y, rcond=None)[0]
        resid0 = y - x @ coeff0
    else:
        coeff0 = np.empty(0)
        resid0 = y.copy()
    scale0 = np.std(resid0)
    if not np.isfinite(scale0) or scale0 <= 0:
        scale0 = 1.0
    theta0 = np.concatenate([coeff0, [np.log(scale0), np.log(2.0)]])

    def neg_ll(theta):
        ncoef = x.shape[1]
        coeff = theta[:ncoef]
        scale = np.exp(theta[ncoef])
        beta = np.exp(theta[ncoef + 1])
        if not np.isfinite(scale) or not np.isfinite(beta) or scale <= 0 or beta <= 0:
            return np.inf
        resid = y - x @ coeff if ncoef > 0 else y
        ll = stats.gennorm.logpdf(resid, beta=beta, loc=0.0, scale=scale).sum()
        return -ll if np.isfinite(ll) else np.inf

    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("optimization failed: " + res.message)
    ncoef = x.shape[1]
    coeff = res.x[:ncoef]
    scale = np.exp(res.x[ncoef])
    beta = np.exp(res.x[ncoef + 1])
    resid = y - x @ coeff if ncoef > 0 else y
    llf = -res.fun
    npar = ncoef + 2
    nfit = len(y)
    aic = 2 * npar - 2 * llf
    bic = np.log(nfit) * npar - 2 * llf
    return {"nar": nar, "trend": trend, "params": coeff, "scale": scale,
        "beta": beta, "resid": resid, "llf": llf, "aic": aic, "bic": bic,
        "nobs_fit": nfit, "prev_level": prev_level}

def best_ar_ged(ts, min_ar_order=1, max_ar_order=5, trend="c"):
    """Select the best GED AR fit by AIC and BIC."""
    best_aic = None
    best_bic = None
    for nar in range(min_ar_order, max_ar_order + 1):
        try:
            fit = fit_ar_ged(ts, nar=nar, trend=trend)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        if best_aic is None or fit["aic"] < best_aic["aic"]:
            best_aic = fit
        if best_bic is None or fit["bic"] < best_bic["bic"]:
            best_bic = fit
    return {"aic": best_aic, "bic": best_bic}

def print_fit_summary(label, fit):
    """Print a short summary of a selected GED AR fit."""
    if fit is None:
        return
    print(f"\nbest {label} lag:", fit["nar"], f"{label}:", fit[label.lower()])
    print("GED beta:", f"{fit['beta']:.3f}", "scale:", f"{fit['scale']:.3f}")
    print("AR coeff:", fit["params"])
