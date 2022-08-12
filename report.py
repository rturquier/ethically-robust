#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code used for the internship report

Required files: Dataset_Stata11compatible.dta (Drupp et al. 2018)


-- References --
Drupp, Moritz A., Freeman, Mark C., Groom, Ben, and Nesje, Frikk. Replication
data for: Discounting Disentangled. Nashville, TN: American Economic
Association [publisher], 2018. Ann Arbor, MI: Inter-university Consortium for
Political and Social Research [distributor], 2019-10-13.
https://doi.org/10.3886/E114692V1
"""


# %% =======================   Imports   =========================
import numpy as np
import pandas as pd
import scipy.stats as stats
import altair as alt


# %% ======================= Functions =========================
def x_axis_from_series(series, n_steps_min=200):
    series = series.dropna()
    max_value = series.max()
    axis = np.linspace(0, max_value, n_steps_min)
    axis = sorted(set(axis) | set(series))

    assert(set(series).issubset(axis))

    return axis


def merge_value_frequency(df_x, col_x, df_y, col_y):
    """
    Add frequencies of ``df_x`` values observed in ``df_y``.

    Parameters
    ----------
    df_x : DataFrame
        DataFrame containing a list of unique values.
    col_x : str
        Name of the column of ``df_x`` containing a list of unique values.
    df_y : DataFrame
        DataFrame containing a list of observations
    col_y : str
        Name of the column of ``df_y`` containing a list of observations.


    Returns
    -------
    DataFrame
        ``df_x`` with an additional column called "freq_" + ``col_y``,
        containing the frequency of each unique value of ``col_x`` observed in
        ``col_y``.
    """
    df = (
        df_y
        .loc[:, col_y]
        .value_counts(normalize=True)
        .pipe(pd.DataFrame)
        .rename(columns={col_y: "freq_" + col_y})
        .merge(df_x, left_index=True, right_on=col_x, how="right")
        .fillna({"freq_" + col_y: 0})
        .reindex(columns=[col_x, "freq_" + col_y])
    )

    return df


def density_chart(df, x, freq, pdf, bin_step=1, x_format="~f"):
    """Plot a histogram with a fitted probability density function."""

    bar_color  = alt.value("#8FBC8F")
    line_color = alt.value("#008B8B")

    bar_x     = alt.X(x, bin=alt.Bin(step=bin_step), axis=alt.Axis(title=""))
    bar_scale = 'datum.' + freq + ' / ' + str(bin_step)
    bar_y     = alt.Y('sum(freq_scaled):Q', axis=alt.Axis(title=""))

    line_x = alt.X(x,   axis=alt.Axis(title="value", format=x_format))
    line_y = alt.Y(pdf, axis=alt.Axis(title="density"), type="quantitative")


    base = alt.Chart(df)

    bar = (
        base.mark_bar()
        .transform_calculate(freq_scaled=bar_scale)
        .encode(x=bar_x, y=bar_y, color=bar_color)
    )

    line = (
        base.mark_line()
        .encode(x=line_x, y=line_y, color=line_color)
    )

    return bar + line


# %% ======================= Main code ===========================
# %% Import data from Drupp et al. (2018)
drupp_2018_data_path = "114692-V1/data/Dataset_Stata11compatible.dta"
drupp_2018_df = pd.read_stata(drupp_2018_data_path)

drupp_2018_df = (drupp_2018_df
                 .loc[:, ["puretp", "eta"]]
                 .rename(columns={"puretp": "delta"})
                 .dropna(how="all")
                 .assign(delta=lambda x: x.delta / 100)
                )

drupp_2018_df.head()

## %% Format data
df_delta = pd.DataFrame({'x_delta': x_axis_from_series(drupp_2018_df.delta)})
df_eta   = pd.DataFrame({'x_eta':   x_axis_from_series(drupp_2018_df.eta  )})

df_delta = merge_value_frequency(df_delta, "x_delta", drupp_2018_df, "delta")
df_eta   = merge_value_frequency(df_eta, "x_eta", drupp_2018_df, "eta")

df_delta.head()

# %% Correct null values of eta to estimate the distribution of Eta
# 0 is not in the support of the gamma distribution.
# Null values of eta are replaced with an arbitrarily low value.
epsilon = 1E-9

drupp_2018_df = drupp_2018_df.assign(
    eta=lambda x: np.where(x.eta == 0, epsilon, x.eta)
)

# %% Estimate parameters with maximum likelihood estimator (MLE)
beta_prime_MLE = (
    stats.betaprime
    .fit(drupp_2018_df.loc[:, "delta"], fix_a=1, floc=0, fscale=1)
    [1]
)

gamma_MLE_shape, gamma_MLE_loc, gamma_MLE_scale = (
    stats.gamma
    .fit(drupp_2018_df.loc[:, "eta"].dropna(), floc=0)
)

fit_alpha = gamma_MLE_shape
fit_beta = 1 / gamma_MLE_scale

print(beta_prime_MLE, fit_alpha, fit_beta, sep="\n")

# %% Estimate parameters with method of moments (MM)
# Beta prime distribution
m = drupp_2018_df.delta.mean()
beta_prime_MM = 1 / m + 1

# Gamma distribution
mu     = drupp_2018_df.eta.mean()
sigma2 = drupp_2018_df.eta.var()

gamma_MM_shape = mu**2  / sigma2
gamma_MM_scale = sigma2 / mu

# %% Add fitted curves to chart dataframes
df_delta = df_delta.assign(
    pdf_delta_MLE=lambda r: stats.betaprime(1, beta_prime_MLE).pdf(r.x_delta),
    pdf_delta_MM =lambda r: stats.betaprime(1, beta_prime_MM ).pdf(r.x_delta)
)

df_eta = df_eta.assign(
    pdf_eta_MLE=lambda r:
        stats.gamma(a=gamma_MLE_shape, scale=gamma_MLE_scale).pdf(r.x_eta),
    pdf_eta_MM=lambda r:
        stats.gamma(a=gamma_MM_shape, scale=gamma_MM_scale).pdf(r.x_eta)
)


# %% Density charts - delta
(
    density_chart(df_delta, x="x_delta", freq="freq_delta",
                  pdf="pdf_delta_MLE", bin_step=0.004, x_format="%"
                 )
    .properties(title={"text": "Distribution of beliefs over \u03b4",
                       "subtitle": "Fit with maximum likelihood estimation"})
    .save("charts/delta_MLE.html")
)


(
    density_chart(df_delta, x="x_delta", freq="freq_delta",
                  pdf="pdf_delta_MM", bin_step=0.004, x_format="%"
                 )
    .properties(title={"text": "Distribution of beliefs over \u03b4",
                       "subtitle": "Fit with method of moments"})
    .save("charts/delta_MM.html")
)


# %% Density charts - eta
(
    density_chart(df_eta, x="x_eta", freq="freq_eta", pdf="pdf_eta_MLE",
                  bin_step=0.4)
    .properties(title={"text": "Distribution of beliefs over \u03b7",
                       "subtitle": "Fit with maximum likelihood estimation"})
    .save("charts/eta_MLE.html")
)


(
    density_chart(df_eta, x="x_eta", freq="freq_eta", pdf="pdf_eta_MM",
                  bin_step=0.4)
    .properties(title={"text": "Distribution of beliefs over \u03b7",
                       "subtitle": "Fit with method of moments"})
    .save("charts/eta_MM.html")
)
