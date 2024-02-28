#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code used for the section on population ethics

Required files: PopEthics Study 3c.csv (Caviola et al. 2022)


-- References --
Caviola, Lucius, David Althaus, Andreas Mogensen, and Geoffrey Goodwin.
Supplemental materials for "Population Ethical Intuitions", 19 January
2021. https://osf.io/qt65w/.
"""

# %% Imports
import pandas as pd
import numpy as np
import portion as intervals
from numbers import Number
import altair as alt


# %% ======= Functions =======
def get_beta_from_indifference(population, utility):
    initial_utility = 100
    initial_population = 1_000
    beta_indifferent = (  np.log(initial_utility / utility)
                        / np.log(population / initial_population))
    return beta_indifferent


@np.vectorize
def get_beta_interval_from_answer(answer, population, utility):
    beta_indifferent = get_beta_from_indifference(population, utility)
    
    if answer < 4:
        preference = "initial"
    elif answer == 4:
        preference = "indifferent"
    elif answer > 4:
        preference = "current"
    
    if preference == "initial":
        beta_min = 0
        beta_max = beta_indifferent
    elif preference == "indifferent":
        beta_min = beta_max = beta_indifferent
    elif preference == "current":
        beta_min = beta_indifferent
        beta_max = 1
    
    return intervals.closed(beta_min, beta_max)


@np.vectorize
def get_midpoint(interval:intervals.Interval) -> float:
    if interval.empty:
        midpoint = np.nan
    else:
        midpoint = (interval.upper + interval.lower) / 2
    return midpoint


def infer_beta(study_3c_df: pd.DataFrame) -> pd.DataFrame:
    processed_df = (
        study_3c_df
        .query("valence == 'happy' & thinking == 'reflection'")
        .assign(
            beta_interval_t_90 = lambda x:
                get_beta_interval_from_answer(x.t_90, 100_000, 90),
            beta_interval_t_70 = lambda x:
                get_beta_interval_from_answer(x.t_70, 100_000, 70),
            beta_interval_t_50k = lambda x:
                get_beta_interval_from_answer(x.t_50k, 100_000, 50),
        )
    )

    processed_df['beta_interval_intersection'] = (
        processed_df.apply(lambda row:
                           row.beta_interval_t_90
                           & row.beta_interval_t_70
                           & row.beta_interval_t_50k,
                           axis=1)
    )

    processed_df['intersection_is_empty'] = (
        processed_df.apply(lambda row: row.beta_interval_intersection.empty,
                           axis=1)
    )

    processed_df = processed_df.assign(
            beta_midpoint = lambda x: get_midpoint(x.beta_interval_intersection)
    )
    
    return processed_df



# ======= Main code ========
# %% Read data
study_3c_path = "osfstorage-archive/Study 3c/PopEthics Study 3c.csv"
study_3c_df = pd.read_csv(study_3c_path)

# %% Process
processed_df = study_3c_df.pipe(infer_beta)


# %% Histogram
processed_df['beta_intersection_lower'] = (
    processed_df['beta_interval_intersection']
    .apply(lambda x: x.lower if not x.empty else np.nan)
)

processed_df['beta_intersection_upper'] = (
    processed_df['beta_interval_intersection']
    .apply(lambda x: x.upper if not x.empty else np.nan)
)

histogram_df = (
    processed_df[['beta_midpoint', 'beta_intersection_upper', 'beta_intersection_lower']]
    .value_counts()
    .reset_index(name="frequency")
    .sort_values("beta_midpoint")
    .assign(
        interval_length = lambda x: x.beta_intersection_upper
                                    - x.beta_intersection_lower,
        frequency_combined = lambda x: np.where(
            x.interval_length > 0,
            x.frequency + x.frequency.shift(1).fillna(0),
            np.nan
        ),
        density = lambda x: x.frequency_combined / x.interval_length
    )
)

chart_base = (
    alt.Chart(histogram_df)
    .properties(width=550, height=300)
)

histogram = (
    chart_base
    .mark_rect()
    .encode(
        x='beta_intersection_lower',
        x2='beta_intersection_upper',
        y=alt.Y('density', axis=None)
    )
)

labels = (
    chart_base
    .transform_filter("datum.interval_length > 0")
    .mark_text(dy=-8)
    .encode(
        x=alt.X(
            'beta_midpoint',
            axis=alt.Axis(title="Estimated 𝛽",
                          values=histogram_df.beta_intersection_upper.values,
                          format=".2")
        ),
        y=alt.Y('density', axis=None),
        text='frequency_combined'
    )
)
(histogram + labels).configure_view(stroke=None)
# (histogram + labels).configure_view(stroke=None).save("charts/caviola_3c_beta_histogram.svg")
# %%
