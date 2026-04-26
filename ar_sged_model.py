""" fit autoregressive models with skewed GED innovations """
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, polygamma
from scipy.integrate import quad
from statsmodels.tools.numdiff import approx_hess3
from ar_ged_model import ar_design

def numerical_hessian(func, x):
    """Numerical Hessian using statsmodels' higher-order finite differences."""
    return np.asarray(approx_hess3(np.asarray(x, dtype=float), func), dtype=float)

def analytic_hessian(theta, y, x):
    """
    Exact piecewise Hessian of the SGED negative log-likelihood in transformed
    parameters (AR coefficients, log(scale), log(beta), log(xi)).
    The formula is valid away from residuals exactly equal to zero.
    """
    theta = np.asarray(theta, dtype=float)
    ncoef = x.shape[1]
    coeff = theta[:ncoef]
    lam = theta[ncoef]
    eta = theta[ncoef + 1]
    kap = theta[ncoef + 2]
    beta = np.exp(eta)
    xi = np.exp(kap)
    resid = y - x @ coeff if ncoef > 0 else y
    abs_resid = np.abs(resid)
    if np.any(abs_resid <= 1.0e-12):
        raise ValueError("analytic Hessian undefined when residuals are too close to zero")
    d = np.where(resid < 0.0, 1.0, -1.0)
    r = np.log(abs_resid) - lam + d * kap
    q = np.exp(beta * r)
    hess = np.zeros((ncoef + 3, ncoef + 3), dtype=float)
    if ncoef > 0:
        w_bb = beta * q * (beta - 1.0) / (resid * resid)
        hess[:ncoef, :ncoef] = x.T @ (x * w_bb[:, None])
        v_lam = beta * beta * q / resid
        v_kap = -beta * beta * q * d / resid
        v_eta = -beta * q * (beta * r + 1.0) / resid
        hess[:ncoef, ncoef] = x.T @ v_lam
        hess[ncoef, :ncoef] = hess[:ncoef, ncoef]
        hess[:ncoef, ncoef + 1] = x.T @ v_eta
        hess[ncoef + 1, :ncoef] = hess[:ncoef, ncoef + 1]
        hess[:ncoef, ncoef + 2] = x.T @ v_kap
        hess[ncoef + 2, :ncoef] = hess[:ncoef, ncoef + 2]
    sech2 = 1.0 / np.cosh(kap) ** 2
    t = np.exp(-eta)
    hess[ncoef, ncoef] = np.sum(beta * beta * q)
    hess[ncoef, ncoef + 1] = -np.sum(beta * q * (beta * r + 1.0))
    hess[ncoef + 1, ncoef] = hess[ncoef, ncoef + 1]
    hess[ncoef, ncoef + 2] = -np.sum(beta * beta * q * d)
    hess[ncoef + 2, ncoef] = hess[ncoef, ncoef + 2]
    hess[ncoef + 1, ncoef + 1] = len(y) * (digamma(t) / beta +
        polygamma(1, t) / (beta * beta)) + np.sum(beta * q * r *
        (beta * r + 1.0))
    hess[ncoef + 1, ncoef + 2] = np.sum(beta * q * d * (beta * r + 1.0))
    hess[ncoef + 2, ncoef + 1] = hess[ncoef + 1, ncoef + 2]
    hess[ncoef + 2, ncoef + 2] = len(y) * sech2 + np.sum(beta * beta * q)
    return hess

def cov_from_hessian(hess, rel_floor=1.0e-8):
    """Invert a Hessian matrix into a covariance estimate."""
    if hess.ndim != 2 or hess.shape[0] != hess.shape[1]:
        return None
    hess = 0.5 * (hess + hess.T)
    try:
        evals = np.linalg.eigvalsh(hess)
        evecs = np.linalg.eigh(hess)[1]
    except np.linalg.LinAlgError:
        return None
    eval_max = np.max(np.abs(evals))
    if not np.isfinite(eval_max) or eval_max <= 0.0:
        return None
    floor = rel_floor * eval_max
    evals_pos = np.maximum(evals, floor)
    return evecs @ np.diag(1.0 / evals_pos) @ evecs.T

def sged_logpdf(x, beta, xi, scale):
    """
    Fernandez-Steel style skewed GED log-density with location 0.
    xi > 0 controls skew; xi = 1 reduces to GED.
    """
    z = np.asarray(x, dtype=float) / scale
    log_norm = np.log(2.0) - np.log(scale) - np.log(xi + 1.0 / xi)
    z_adj = np.where(z < 0.0, xi * z, z / xi)
    return log_norm + stats.gennorm.logpdf(z_adj, beta=beta, loc=0.0, scale=1.0)

def implied_sged_moments(beta, xi, scale):
    """Return mean, variance, skewness, and excess kurtosis implied by the fit."""
    def pdf_scalar(x):
        return float(np.exp(sged_logpdf(np.array([x]), beta=beta, xi=xi,
            scale=scale))[0])
    def raw_moment(order):
        val, _ = quad(lambda x: (x ** order) * pdf_scalar(x), -np.inf, np.inf,
            limit=400)
        return val
    m1 = raw_moment(1)
    m2 = raw_moment(2)
    m3 = raw_moment(3)
    m4 = raw_moment(4)
    var = m2 - m1 * m1
    mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    mu4 = m4 - 4.0 * m1 * m3 + 6.0 * (m1 ** 2) * m2 - 3.0 * m1 ** 4
    skew = mu3 / (var ** 1.5)
    ex_kurt = mu4 / (var ** 2) - 3.0
    return {"mean": m1, "var": var, "skew": skew,
        "excess_kurtosis": ex_kurt}

def fit_ar_sged(ts, nar=1, trend="c", hessian_method="analytic",
    verify_analytic=False):
    """Fit an AR model by conditional MLE under skewed GED innovations."""
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
    theta0 = np.concatenate([coeff0, [np.log(scale0), np.log(2.0), 0.0]])

    def neg_ll(theta):
        ncoef = x.shape[1]
        coeff = theta[:ncoef]
        scale = np.exp(theta[ncoef])
        beta = np.exp(theta[ncoef + 1])
        xi = np.exp(theta[ncoef + 2])
        if not np.isfinite(scale) or not np.isfinite(beta) or not np.isfinite(xi):
            return np.inf
        if scale <= 0 or beta <= 0 or xi <= 0:
            return np.inf
        resid = y - x @ coeff if ncoef > 0 else y
        ll = sged_logpdf(resid, beta=beta, xi=xi, scale=scale).sum()
        return -ll if np.isfinite(ll) else np.inf

    res = minimize(neg_ll, theta0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError("optimization failed: " + res.message)
    ncoef = x.shape[1]
    hess_theta_analytic = None
    hess_theta_numerical = None
    if hessian_method == "analytic" or verify_analytic:
        hess_theta_analytic = analytic_hessian(res.x, y, x)
    if hessian_method == "numerical" or verify_analytic:
        hess_theta_numerical = numerical_hessian(neg_ll, res.x)
    if hessian_method == "analytic":
        hess_theta = hess_theta_analytic
    elif hessian_method == "numerical":
        hess_theta = hess_theta_numerical
    else:
        raise ValueError("hessian_method must be 'analytic' or 'numerical'")
    cov_theta = cov_from_hessian(hess_theta)
    if cov_theta is not None and cov_theta.shape[0] == len(res.x):
        se_theta = np.sqrt(np.maximum(np.diag(cov_theta), 0.0))
    else:
        se_theta = np.full(len(res.x), np.nan)
    coeff = res.x[:ncoef]
    scale = np.exp(res.x[ncoef])
    beta = np.exp(res.x[ncoef + 1])
    xi = np.exp(res.x[ncoef + 2])
    resid = y - x @ coeff if ncoef > 0 else y
    llf = -res.fun
    npar = ncoef + 3
    nfit = len(y)
    aic = 2 * npar - 2 * llf
    bic = np.log(nfit) * npar - 2 * llf
    se_coeff = se_theta[:ncoef]
    se_log_scale = se_theta[ncoef]
    se_log_beta = se_theta[ncoef + 1]
    se_log_xi = se_theta[ncoef + 2]
    se_scale = scale * se_log_scale
    se_beta = beta * se_log_beta
    se_xi = xi * se_log_xi
    z_log_xi = np.nan
    p_log_xi = np.nan
    if np.isfinite(se_log_xi) and se_log_xi > 0:
        z_log_xi = res.x[ncoef + 2] / se_log_xi
        p_log_xi = 2.0 * stats.norm.sf(abs(z_log_xi))
    implied = implied_sged_moments(beta, xi, scale)
    hess_max_abs_diff = np.nan
    hess_rel_max_abs_diff = np.nan
    if hess_theta_analytic is not None and hess_theta_numerical is not None:
        diff = hess_theta_analytic - hess_theta_numerical
        hess_max_abs_diff = np.max(np.abs(diff))
        denom = np.max(np.abs(hess_theta_numerical))
        if np.isfinite(denom) and denom > 0:
            hess_rel_max_abs_diff = hess_max_abs_diff / denom
    return {"nar": nar, "trend": trend, "params": coeff, "scale": scale,
        "beta": beta, "xi": xi, "resid": resid, "llf": llf, "aic": aic,
        "bic": bic, "nobs_fit": nfit, "prev_level": prev_level,
        "theta": res.x, "hessian_method": hessian_method,
        "hess_theta": hess_theta, "hess_theta_analytic": hess_theta_analytic,
        "hess_theta_numerical": hess_theta_numerical, "cov_theta": cov_theta,
        "se_coeff": se_coeff,
        "se_log_scale": se_log_scale, "se_log_beta": se_log_beta,
        "se_log_xi": se_log_xi, "se_scale": se_scale, "se_beta": se_beta,
        "se_xi": se_xi, "z_log_xi": z_log_xi, "p_log_xi": p_log_xi,
        "implied_mean": implied["mean"], "implied_var": implied["var"],
        "implied_skew": implied["skew"],
        "implied_excess_kurtosis": implied["excess_kurtosis"],
        "hess_max_abs_diff": hess_max_abs_diff,
        "hess_rel_max_abs_diff": hess_rel_max_abs_diff}

def best_ar_sged(ts, min_ar_order=1, max_ar_order=5, trend="c",
    hessian_method="analytic", verify_analytic=False):
    """Select the best skewed GED AR fit by AIC and BIC."""
    best_aic = None
    best_bic = None
    for nar in range(min_ar_order, max_ar_order + 1):
        try:
            fit = fit_ar_sged(ts, nar=nar, trend=trend,
                hessian_method=hessian_method,
                verify_analytic=verify_analytic)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        if best_aic is None or fit["aic"] < best_aic["aic"]:
            best_aic = fit
        if best_bic is None or fit["bic"] < best_bic["bic"]:
            best_bic = fit
    return {"aic": best_aic, "bic": best_bic}

def print_fit_summary(label, fit):
    """Print a short summary of a selected skewed GED AR fit."""
    if fit is None:
        return
    print(f"\nbest {label} lag:", fit["nar"], f"{label}:", fit[label.lower()])
    print("SGED beta:", f"{fit['beta']:.3f}", "xi:", f"{fit['xi']:.3f}",
        "scale:", f"{fit['scale']:.3f}")
    print("AR coeff:", fit["params"])
    print("\nparameter standard errors")
    print(f"{'param':>12}{'estimate':>12}{'stderr':>12}")
    for i, coeff in enumerate(fit["params"]):
        print(f"{('b' + str(i)):>12}{coeff:12.3f}{fit['se_coeff'][i]:12.3f}")
    print(f"{'scale':>12}{fit['scale']:12.3f}{fit['se_scale']:12.3f}")
    print(f"{'beta':>12}{fit['beta']:12.3f}{fit['se_beta']:12.3f}")
    print(f"{'xi':>12}{fit['xi']:12.3f}{fit['se_xi']:12.3f}")
    print("test xi = 1 via log(xi) = 0:",
        "z =", f"{fit['z_log_xi']:.3f}",
        "p =", f"{fit['p_log_xi']:.4g}")
    print("implied noise skew:", f"{fit['implied_skew']:.3f}",
        "implied excess kurtosis:", f"{fit['implied_excess_kurtosis']:.3f}")
    print("Hessian used:", fit["hessian_method"])
    if np.isfinite(fit["hess_max_abs_diff"]):
        print("analytic vs numerical Hessian max abs diff:",
            f"{fit['hess_max_abs_diff']:.3e}",
            "relative:", f"{fit['hess_rel_max_abs_diff']:.3e}")
