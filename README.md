# Time-Series-Distributions

This project fits and compares probability distributions for a univariate time series, optionally after simple preprocessing and autoregressive filtering.

The main use case is:

- read a dated CSV series
- optionally transform it
- optionally fit an `AR(p)` model
- fit multiple candidate distributions to the transformed series or AR residuals
- compare them by log-likelihood, AIC, BIC, and implied skewness and excess kurtosis

The repository is designed for time-series diagnostics rather than for generic i.i.d. data fitting.

## Main script

- `xfit_distributions.py`

This is the entry point. It:

- reads one column from a CSV file
- supports date filtering
- supports log transform, differencing, demeaning, and standardization
- loops over AR orders `p_min` through `p_max`
- fits a collection of one-piece distributions and mixture distributions
- prints sample moment diagnostics and information-criterion comparisons
- prints detailed parameter summaries for each fitted distribution

Interpretation of `p`:

- `p = 0`: fit distributions directly to the preprocessed series
- `p > 0`: fit distributions to residuals from an `AR(p)` model

## Supported distributions

One-piece distributions:

- normal
- GED
- SGED
- NIG
- GH
- Student `t`
- skew-`t`

Mixture distributions:

- normal mixtures
- hyperbolic secant mixtures

The script can loop over component counts for the mixture families.

## Preprocessing options

Configured near the top of `xfit_distributions.py`:

- `take_logs`
- `take_diff`
- `demean`
- `standardize`
- `p_min`, `p_max`
- `ar_trend`

Typical workflow:

1. choose a series and date range
2. decide whether to work in levels or logs
3. decide whether to difference
4. decide whether to fit the series directly or AR residuals
5. compare candidate distributions by AIC/BIC

## Output

For each AR order, the script prints:

- sample `mean`, `sd`, `skew`, and `excess kurtosis`
- rough standard errors for those moments
- a distribution comparison table with:
  - number of parameters
  - log-likelihood
  - AIC
  - BIC
  - AIC rank
  - BIC rank
  - implied skew
  - implied excess kurtosis
- detailed parameter summaries for selected fits

At the end it prints the best-fitting distribution by BIC for each AR order.

## Files

- `xfit_distributions.py`
  Main entry point.

- `sged_fit.py`
  Standalone normal, GED, and SGED fitting and reporting helpers.

- `nig_fit.py`
  Normal-inverse-Gaussian fitting and reporting.

- `gh_fit.py`
  Generalized hyperbolic fitting and reporting. Includes a constrained option to keep the fit away from problematic parameter regions.

- `t_fit.py`
  Student `t` and skew-`t` fitting and reporting.

- `normal_mix_fit.py`
  Normal-mixture fitting and moment calculations.

- `hypsec_mix_fit.py`
  Hyperbolic-secant-mixture fitting and moment calculations.

- `dist_fit_util.py`
  Shared helper utilities for comparison tables.

- `pandas_util.py`
  Minimal CSV/date-index helpers used by the project.

## Example

The default configuration uses:

- `infile = "vix.csv"`
- `take_logs = True`
- `p_min = p_max = 1`

So by default the script fits distributions to `AR(1)` residuals of `log(VIX)`.

Run:

```bash
python xfit_distributions.py
```

## Results from `results.txt`

The included `results.txt` shows the default run on `vix.csv` with:

- `take_logs = True`
- `p_min = p_max = 1`
- `ar_trend = "c"`

So the script fits candidate distributions to the residuals of an `AR(1)` model for `log(VIX)`.

### AR(1) benchmark

The fitted AR(1) is:

- intercept: `0.0579`
- AR coefficient: `0.9801`
- AR log-likelihood: `11630.297`

The residuals are clearly non-Gaussian:

- residual sd: `0.068`
- residual skew: `1.057`
- residual excess kurtosis: `6.806`

So a Gaussian residual law is a weak benchmark rather than a plausible final model.

### Distribution comparison

The default run compares one-piece and mixture distributions on these residuals.

Key BIC values from `results.txt` are:

| distribution | BIC |
|---|---:|
| normal | -23251.472 |
| GED | -24540.051 |
| SGED | -24531.944 |
| NIG | -24657.218 |
| GH | -24566.119 |
| t | -24696.485 |
| skew-t | -24687.605 |
| mix2 | -24694.515 |
| mix3 | -24672.309 |
| mix4 | -24732.744 |
| hypsec2 | -24825.580 |
| hypsec3 | -24821.536 |
| hypsec4 | -24796.960 |

The best fit by BIC is:

- `hypsec2` with BIC `-24825.580`

and the best fit by AIC is:

- `hypsec3` with AIC `-24878.503`

So the leading models are hyperbolic-secant mixtures rather than one-piece distributions.

### Interpretation

Several patterns stand out.

First, every heavy-tailed candidate beats the normal distribution by a large margin. This confirms that the AR(1) residuals of `log(VIX)` are far from Gaussian.

Second, simple one-piece skewed distributions help, but not enough:

- GED and SGED improve strongly on normal
- NIG improves further
- constrained GH does not perform especially well in this example

Third, the best likelihood-based fits come from mixture models, especially hyperbolic-secant mixtures. That suggests the residual law is better described as a mixture than as a single smooth family.

### Best-fitting model in the default run

The preferred BIC model, `hypsec2`, has:

- implied skew: `1.102`
- implied excess kurtosis: `5.955`

which is close to the empirical residual moments:

- empirical skew: `1.057`
- empirical excess kurtosis: `6.806`

Its two fitted components are:

| component | weight | mean | sd |
|---|---:|---:|---:|
| 1 | 0.906 | -0.008 | 0.036 |
| 2 | 0.094 | 0.071 | 0.071 |

This is a compact description of the residual distribution:

- a dominant low-variance near-zero component
- a smaller positive-shifted higher-variance component

That small positive, higher-volatility component is enough to generate substantial positive skewness and excess kurtosis.

### One-piece versus mixture fits

The `t` and `skew-t` fits achieve good AIC/BIC values, but in the default run they sit on the `df = 4` boundary and imply absurdly large excess kurtosis. So they should be interpreted cautiously.

By contrast, `hypsec2` gives:

- better BIC than `t`, `skew-t`, `NIG`, `GED`, and `SGED`
- implied moments that are much more plausible relative to the sample residual moments

For this example, that makes the 2-component hyperbolic-secant mixture the most credible overall choice.

## Notes

- This repo is univariate.
- It focuses on distribution comparison for transformed time-series data and AR residuals.
- Some families are zero-location parameterizations, so fitting them to raw non-centered data may be less meaningful than fitting them to demeaned data or residuals.
- The GH fitter defaults to a constrained version, because unconstrained fits can drift into pathological heavy-tail regions.

## Dependencies

The code uses:

- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `scikit-learn`

## Summary

The main idea of the project is simple: before assuming Gaussian residuals, fit a range of alternative distributions directly to the transformed series or AR residuals and compare them systematically.
