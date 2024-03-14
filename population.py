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
import scipy.stats as stats

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


def intersect_two(left:pd.Series, right:pd.Series) -> pd.Series:
    df = pd.DataFrame({'left': left, 'right': right})
    intersection = df.apply(lambda row: row['left'] & row['right'], axis=1)
    return intersection


def intersect(df, interval_column_names:list) -> pd.Series:
    for i, column_name in enumerate(interval_column_names):
        if i == 0:
            left_column = df[column_name]
            continue
        elif i == 1:
            right_column = df[column_name]
            intersection = intersect_two(left_column, right_column)
        else:
            right_column = df[column_name]
            intersection = intersect_two(intersection, right_column)
    return intersection


def create_intervals(df):
    df = df.assign(
        beta_interval_90k = lambda x:
            get_beta_interval_from_answer(x.t_90, 100_000, 90),
        beta_interval_70k = lambda x:
            get_beta_interval_from_answer(x.t_70, 100_000, 70),
        beta_interval_50k = lambda x:
            get_beta_interval_from_answer(x.t_50k, 100_000, 50),
        beta_interval_50m = lambda x:
            get_beta_interval_from_answer(x.t_50m, 1_000_000, 50),
        beta_interval_50b = lambda x:
            get_beta_interval_from_answer(x.t_50b, 1_000_000_000, 50)
    )
    return df


def get_intersections_and_bounds_and_midpoints(df, question_subsets:dict):
    for prefix, column_names in question_subsets.items():
        df[prefix + '_intersection'] = df.pipe(intersect, column_names)
        
        df[prefix + '_lower'] = (
            df[prefix + '_intersection']
            .apply(lambda x: x.lower if not x.empty else np.nan)
        )
        
        df[prefix + '_upper'] = (
            df[prefix + '_intersection']
            .apply(lambda x: x.upper if not x.empty else np.nan)
        )
        
        df[prefix + '_midpoint'] = (
            df[prefix + '_intersection'].pipe(get_midpoint)
        )
    return df


def process(study_3c_df: pd.DataFrame) -> pd.DataFrame:
    interval_columns = ["beta_interval_90k",
                        "beta_interval_70k",
                        "beta_interval_50k",
                        "beta_interval_50m",
                        "beta_interval_50b"]
    question_subsets = {'all': interval_columns,
                        '90_70_50': interval_columns[:3],
                        'k_m_b': interval_columns[-3:]}
    
    processed_df = (
        study_3c_df
        .query("valence == 'happy' & thinking == 'reflection'")
        .pipe(create_intervals)
        .pipe(get_intersections_and_bounds_and_midpoints, question_subsets)
        .drop(columns=interval_columns)
    )
    return processed_df


def carry_single_point_values_to_next_bin(df) -> pd.Series: 
    current_line_is_single_point = (df.interval_length == 0)
    previous_line = df.shift(1, fill_value=0)
    previous_line_is_single_point = (previous_line.interval_length == 0)
    
    frequency_combined = np.select(
        [current_line_is_single_point, previous_line_is_single_point, True],
        [np.nan, df.frequency + previous_line.frequency, df.frequency]
    )
    return frequency_combined


def prepare_histogram_df(processed_df:pd.DataFrame, prefix:str):
    useful_columns_suffixes = ["_midpoint", "_upper", "_lower"]
    useful_columns = [prefix + suffix for suffix in useful_columns_suffixes]
    
    histogram_df = (
        processed_df[useful_columns]
        .value_counts()
        .reset_index(name="frequency")
        .sort_values(prefix + "_midpoint")
        .assign(
            interval_length = lambda x: x[prefix + '_upper']
                                        - x[prefix + '_lower'],
            frequency_combined = carry_single_point_values_to_next_bin,
            density = lambda x: x.frequency_combined / x.interval_length
        )
    )
    return histogram_df    
    

def make_beta_histogram(processed_df:pd.DataFrame, prefix:str) -> alt.Chart:
    histogram_df = prepare_histogram_df(processed_df, prefix)
    
    chart_base = (
        alt.Chart(histogram_df)
        .properties(width=550, height=300)
    )

    histogram = (
        chart_base
        .mark_rect()
        .encode(
            x=prefix + '_lower',
            x2=prefix + '_upper',
            y=alt.Y('density', axis=None)
        )
    )

    label_values_except_zero = histogram_df[prefix + '_upper'].values
    label_values = np.insert(label_values_except_zero, 0, 0)
    labels = (
        chart_base
        .mark_text(dy=-8)
        .encode(
            x=alt.X(
                prefix + '_midpoint',
                axis=alt.Axis(title="Estimated 𝛽",
                              values=label_values,
                              labelOverlap="greedy",
                              format=".2")
            ),
            y=alt.Y('density', axis=None),
            text='frequency_combined'
        )
    )
    
    return (histogram + labels).configure_view(stroke=None)


# ======= Main code ========
# %% Read data
study_3c_path = "osfstorage-archive/Study 3c/PopEthics Study 3c.csv"
study_3c_df = pd.read_csv(study_3c_path)

# %% Process
processed_df = process(study_3c_df)

# %% Histograms
make_beta_histogram(processed_df, "all")
make_beta_histogram(processed_df, "90_70_50")
make_beta_histogram(processed_df, "k_m_b")


# %% Calibrate
class kumaraswamy_gen(stats.rv_continuous):
    "Kumaraswamy distribution"
    def _pdf(self, x, a, b):
        return a * b * x**(a - 1) * (1 - x**a)**(b-1)

import matplotlib.pyplot as plt
kumaraswamy_frozen = kumaraswamy_gen(a=0.5, b=1)
a = 0.5
b = 1
fig, ax = plt.subplots(1, 1)
x = np.linspace(0, 1, 100)
ax.plot(x, kumaraswamy_frozen.pdf(x, a, b),
       'r-', lw=5, alpha=0.6, label='Kumaraswamy pdf')
fig


kumaraswamy_gen().fit(processed_df.loc[:, "all_midpoint"].dropna())




stats.beta.fit(processed_df.loc[:, "all_midpoint"].dropna())