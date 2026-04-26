""" fit and compare multiple distributions for a univariate data series

Preprocessing options include:
- optional log transform
- optional first difference
- optional demeaning
- optional standardization

The script can also loop over AR orders `p_min` through `p_max`:
- `p = 0`: fit distributions to the preprocessed series itself
- `p > 0`: fit an AR(p) model to the preprocessed series and then fit
  distributions to the AR residuals
"""
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import statsmodels.api as sm

from dist_fit_util import count_fit_params
from gh_fit import fit_gh, print_fit_summary as print_gh_fit_summary
from hypsec_mix_fit import fit_hypsec_mix, print_fit_summary as print_hypsec_mix_summary
from nig_fit import fit_nig, print_fit_summary as print_nig_fit_summary
from normal_mix_fit import fit_normal_mix, print_fit_summary as print_mix_fit_summary
from pandas_util import print_first_last, read_csv_date_index
from sged_fit import fit_ged, fit_normal, fit_sged, print_fit_summary
from t_fit import fit_skewt, fit_t, print_fit_summary as print_t_fit_summary


infile = "vix.csv"
date_min = date(1900, 1, 1)
date_max = date(2100, 12, 31)
col_use = 0
take_logs = True
p_min = 1
p_max = 1
ar_trend = "c"
take_diff = False
demean = False
standardize = False

do_fit_ged_sged = True
do_fit_nig = True
do_fit_gh = True
gh_constrained = True
do_fit_t = True
do_fit_normal_mix = True
normal_mix_min_components = 2
normal_mix_max_components = 4
do_fit_hypsec_mix = True
hypsec_mix_min_components = 2
hypsec_mix_max_components = 4

print("data file:", infile)
print("date_min:", date_min)
print("date_max:", date_max)
print("take_logs:", take_logs)
print("take_diff:", take_diff)
print("p_min:", p_min)
print("p_max:", p_max)
print("ar_trend:", ar_trend)
print("demean:", demean)
print("standardize:", standardize)
print("fit_ged_sged:", do_fit_ged_sged)
print("fit_nig:", do_fit_nig)
print("fit_gh:", do_fit_gh)
print("gh_constrained:", gh_constrained)
print("fit_t:", do_fit_t)
print("fit_normal_mix:", do_fit_normal_mix)
print("normal_mix_min_components:", normal_mix_min_components)
print("normal_mix_max_components:", normal_mix_max_components)
print("fit_hypsec_mix:", do_fit_hypsec_mix)
print("hypsec_mix_min_components:", hypsec_mix_min_components)
print("hypsec_mix_max_components:", hypsec_mix_max_components)

df = read_csv_date_index(infile, date_min=date_min, date_max=date_max)
df = df.apply(pd.to_numeric, errors="coerce").astype(float)
ser = df.iloc[:, col_use].dropna()
if take_logs:
    ser_use = np.log(ser.where(ser > 0)).dropna()
    series_title = f"\nlog({ser.name})"
else:
    ser_use = ser.copy()
    series_title = f"\n{ser.name}"
if take_diff:
    ser_use = ser_use.diff().dropna()
    series_title = f"{series_title} diff"
if demean:
    ser_use = ser_use - ser_use.mean()
if standardize:
    ser_use = ser_use / ser_use.std(ddof=1)
ser_model = pd.Series(ser_use.to_numpy(), name=ser.name)

print_first_last(ser_use, title=series_title)
print(ser_use.describe())

best_rows = []

for ar_order in range(p_min, p_max + 1):
    print(f"\n{'=' * 72}")
    print("AR order:", ar_order)
    if ar_order == 0:
        x = ser_model.to_numpy()
        x_title = "series"
    else:
        mod = sm.tsa.AutoReg(ser_model, lags=ar_order, trend=ar_trend, old_names=False)
        ar_res = mod.fit()
        print("\nAR fit summary\n")
        print(ar_res.summary())
        x = np.asarray(ar_res.resid, dtype=float)
        x = x[np.isfinite(x)]
        x_title = f"AR({ar_order}) residuals"

    nobs = len(x)
    se_mean = np.std(x, ddof=1) / np.sqrt(nobs)
    se_sd = np.std(x, ddof=1) / np.sqrt(2.0 * max(nobs - 1, 1))
    se_skew = np.sqrt(6.0 / nobs)
    se_ex_kurt = np.sqrt(24.0 / nobs)

    print(f"\n{x_title} stats")
    print(f"{'':>8}{'mean':>10}{'sd':>10}{'skew':>10}{'ex kurt':>10}")
    print(
        f"{'est':>8}{np.mean(x):10.3f}{np.std(x, ddof=1):10.3f}"
        f"{skew(x, bias=False):10.3f}{kurtosis(x, fisher=True, bias=False):10.3f}"
    )
    print(
        f"{'se':>8}{se_mean:10.3f}{se_sd:10.3f}"
        f"{se_skew:10.3f}{se_ex_kurt:10.3f}"
    )

    fit_rows = []
    detail_rows = []

    normal_fit = fit_normal(x)
    fit_rows.append(("normal", normal_fit))

    if do_fit_ged_sged:
        ged_fit = fit_ged(x)
        sged_fit = fit_sged(x)
        fit_rows.append(("GED", ged_fit))
        fit_rows.append(("SGED", sged_fit))
        detail_rows.append(("GED fit", ged_fit, print_fit_summary))
        detail_rows.append(("SGED fit", sged_fit, print_fit_summary))

    if do_fit_nig:
        nig_fit = fit_nig(x)
        fit_rows.append(("NIG", nig_fit))
        detail_rows.append(("NIG fit", nig_fit, print_nig_fit_summary))

    if do_fit_gh:
        gh_fit = fit_gh(x, constrained=gh_constrained)
        fit_rows.append(("GH", gh_fit))
        detail_rows.append(("GH fit", gh_fit, print_gh_fit_summary))

    if do_fit_t:
        t_fit = fit_t(x)
        skewt_fit = fit_skewt(x)
        fit_rows.append(("t", t_fit))
        fit_rows.append(("skew-t", skewt_fit))
        detail_rows.append(("t fit", t_fit, print_t_fit_summary))
        detail_rows.append(("skew-t fit", skewt_fit, print_t_fit_summary))

    if do_fit_normal_mix:
        for n_components in range(normal_mix_min_components,
                normal_mix_max_components + 1):
            mix_fit = fit_normal_mix(x, n_components=n_components)
            fit_rows.append((f"mix{n_components}", mix_fit))
            detail_rows.append((
                f"normal mixture fit ({n_components} components)",
                mix_fit,
                print_mix_fit_summary,
            ))

    if do_fit_hypsec_mix:
        for n_components in range(hypsec_mix_min_components,
                hypsec_mix_max_components + 1):
            hypsec_fit = fit_hypsec_mix(x, n_components=n_components)
            fit_rows.append((f"hypsec{n_components}", hypsec_fit))
            detail_rows.append((
                f"hyperbolic secant mixture fit ({n_components} components)",
                hypsec_fit,
                print_hypsec_mix_summary,
            ))

    print("\ndistribution comparison")
    fit_df = pd.DataFrame([
        {"dist": dist_name, **dist_fit} for dist_name, dist_fit in fit_rows
    ])
    fit_df["aic_rank"] = fit_df["aic"].rank(method="min").astype(int)
    fit_df["bic_rank"] = fit_df["bic"].rank(method="min").astype(int)
    print(
        f"{'dist':>10}{'#par':>8}{'loglik':>14}{'AIC':>14}{'BIC':>14}"
        f"{'AIC_rank':>10}{'BIC_rank':>10}"
        f"{'skew':>10}{'ex kurt':>12}"
    )
    best_name = None
    best_bic = np.inf
    for dist_name, dist_fit in fit_rows:
        npar = count_fit_params(dist_name, dist_fit)
        row = fit_df.loc[fit_df["dist"] == dist_name].iloc[0]
        print(
            f"{dist_name:>10}{int(npar):8d}{dist_fit['llf']:14.3f}"
            f"{dist_fit['aic']:14.3f}{dist_fit['bic']:14.3f}"
            f"{int(row['aic_rank']):10d}{int(row['bic_rank']):10d}"
            f"{dist_fit['implied_skew']:10.3f}{dist_fit['implied_excess_kurtosis']:12.3f}"
        )
        if dist_fit["bic"] < best_bic:
            best_bic = dist_fit["bic"]
            best_name = dist_name

    best_rows.append({"p": ar_order, "dist": best_name, "bic": best_bic})

    for label, fit, print_func in detail_rows:
        print_func(label, fit)

print("\nbest fit by p")
print(f"{'p':>6}{'dist':>12}{'BIC':>14}")
for row in best_rows:
    print(f"{row['p']:6d}{row['dist']:>12}{row['bic']:14.3f}")
