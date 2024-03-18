#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse survey data from Drupp et al. (2018a)

Required files: Dataset_Stata11compatible.dta (Drupp et al. 2018b)


-- References --
Drupp, Moritz A., Mark C. Freeman, Ben Groom, and Frikk Nesje. 2018a.
'Discounting Disentangled'. American Economic Journal: Economic Policy.
https://doi.org/10.1257/pol.20160240.

Drupp, Moritz A., Freeman, Mark C., Groom, Ben, and Nesje, Frikk. 2018b.
Replication data for: Discounting Disentangled.
https://doi.org/10.3886/E114692V1
"""

# %% Imports
import numpy as np
import pandas as pd
import scipy.stats as stats
import altair as alt

import functions as f


# %% Read data from Drupp et al. (2018b)
drupp_2018_data_path = "114692-V1/data/Dataset_Stata11compatible.dta"
drupp_2018_df = pd.read_stata(drupp_2018_data_path)

drupp_2018_df = (drupp_2018_df
                 .loc[:, ["puretp", "eta"]]
                 .rename(columns={"puretp": "delta"})
                 .dropna(how="all")
                 .assign(delta=lambda x: x.delta / 100)
                )

drupp_2018_df.head()

# %% Prepare dataframes for the charts
df_delta = pd.DataFrame({'x_delta': f.x_axis_from_series(drupp_2018_df.delta)})
df_eta   = pd.DataFrame({'x_eta':   f.x_axis_from_series(drupp_2018_df.eta  )})

df_delta = f.merge_value_frequency(df_delta, "x_delta", drupp_2018_df, "delta")
df_eta   = f.merge_value_frequency(df_eta, "x_eta", drupp_2018_df, "eta")

df_delta.head()

# %% Correct null values of eta to estimate the distribution of Eta
# 0 is not in the support of the gamma distribution.
# Null values of eta are replaced with an arbitrarily low value.
epsilon = 1E-15

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
beta_prime_MM = 1 + 1 / m

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
    f.density_chart(df_delta, x="x_delta", freq="freq_delta",
                  pdf="pdf_delta_MLE", bin_step=0.005, x_format="~%"
                 )
    # .properties(title={"text": "Distribution of beliefs over \u03b4",
    #                    "subtitle": "Fit with maximum likelihood estimation"})
    .save("charts/delta_MLE.svg")
)


(
    f.density_chart(df_delta, x="x_delta", freq="freq_delta",
                  pdf="pdf_delta_MM", bin_step=0.005, x_format="~%"
                 )
    # .properties(title={"text": "Distribution of beliefs over \u03b4",
    #                    "subtitle": "Fit with method of moments"})
    .save("charts/delta_MM.svg")
)


# %% Density charts - eta
(
    f.density_chart(df_eta, x="x_eta", freq="freq_eta", pdf="pdf_eta_MLE",
                  bin_step=0.5)
    .properties(title={"text": "Distribution of beliefs over \u03b7",
                       "subtitle": "Fit with maximum likelihood estimation"})
    .save("charts/eta_MLE.svg")
)


(
    f.density_chart(df_eta, x="x_eta", freq="freq_eta", pdf="pdf_eta_MM",
                  bin_step=0.5)
    .properties(title={"text": "Distribution of beliefs over \u03b7",
                       "subtitle": "Fit with method of moments"})
    .save("charts/eta_MM.svg")
)


# %% Prepare the dataframe with the social discount rate (SDR)
favorite_delta = df_delta.set_index('x_delta').freq_delta.idxmax()
favorite_eta = df_eta.set_index('x_eta').freq_eta.idxmax()

df_sdr = (
    pd.DataFrame()
    .assign(
        year=np.linspace(1, 500, 300),
        sdr_uncertain_approx=lambda r: f.ramsey_beta_prime_gamma(
            tau=r.year, g=0.02, beta_prime=beta_prime_MM,
            alpha=gamma_MM_shape, beta=1 / gamma_MM_scale
        ),
        sdr_uncertain_exact=lambda r: f.exact_sdr_beta_prime_gamma(
            tau=r.year, g=0.02, beta_prime=beta_prime_MM,
            alpha=gamma_MM_shape, beta=1 / gamma_MM_scale
        ),
        sdr_uncertain_error=lambda r:
            (r.sdr_uncertain_exact - r.sdr_uncertain_approx)
            / r.sdr_uncertain_approx,
        sdr_certain_ramsey=f.ramsey(delta=m, eta=mu, g=0.02),
        sdr_favorite_theory=f.ramsey(
            delta=favorite_delta, eta=favorite_eta, g=0.02
        ),
        sdf_uncertain=lambda r:
            f.sdr_to_sdf(r.sdr_uncertain_exact, r.year),
        sdf_certain=lambda r:
            f.sdr_to_sdf(r.sdr_certain_ramsey, r.year),
        sdf_ratio=lambda r: r.sdf_uncertain / r.sdf_certain,
        sdf_favorite_theory=lambda r:
            f.sdr_to_sdf(r.sdr_favorite_theory, r.year),
    )
)

# %% Plot relative approximation error
(
    df_sdr
    .pipe(f.line_chart, x="year", y="sdr_uncertain_error",
          x_title="Year",
          y_title="Relative approximation error", y_format="%")
    .properties(width=600, height=300)
    .save("charts/relative_approximation_error.svg")
)

# %% Plot of social discount rate
legend_dict = {
    "sdr_certain_ramsey": "Average values",
    "sdr_favorite_theory": "Modal values",
    "sdr_uncertain_approx": "Expected choice-worthiness"
}

(
    df_sdr
    .loc[:, [
        "year",
        "sdr_uncertain_approx",
        "sdr_certain_ramsey",
        "sdr_favorite_theory",
    ]]
    .melt("year")
    .pipe(
        f.line_chart, x="year", y="value",
        color="variable",
        strokeDash="variable",
        multi=True,
        # title = "Long-run social discount rate",
        legend=alt.Legend(
            title=None,
            labelExpr=f.dict_to_labelExpr(legend_dict),
            orient="bottom"
        ),
        x_title="Year",
        y_title="Social discount rate",
        y_format="%"
    )
    .properties(width=600, height=300)
    .save("charts/social_discount_rate.svg")
)

# %% Plot social discount factor
legend_dict = {
    "sdf_uncertain": "Expected choice-worthiness",
    "sdf_certain": "Average values",
    "sdf_favorite_theory": "Modal values",
}

(
    df_sdr
    .loc[:, ["year", "sdf_uncertain", "sdf_certain", "sdf_favorite_theory"]]
    .melt("year")
    .pipe(
        f.line_chart, x="year", y="value",
        multi=True,
        color="variable",
        strokeDash="variable",
        # title = "Long-run social discount factor",
        legend=alt.Legend(
            title=None,
            labelExpr=f.dict_to_labelExpr(legend_dict),
            orient="bottom"
        ),
        x_title="Year",
        y_title="Social discount factor",
        # y_scale=alt.Scale(type="log"),
        y_format="%"
    )
    .properties(width=600, height=300)
    .save("charts/social_discount_factor.svg")
)

# %% plot ratio of factors
(
    df_sdr
    .pipe(
        f.line_chart, x="year", y="sdf_ratio",
        y_scale=alt.Scale(type="log"),
        x_title="Year",
        y_title="Morally uncertain discount factor over standard factor",
        color="#b12447"
    )
    .properties(width=600, height=300)
    .save("charts/discount_factor_ratio.svg")
)
