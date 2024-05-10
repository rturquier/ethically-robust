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
import altair as alt
import scipy as sp
import requests

import functions as f

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


def infer_beta_from_answers(study_3c_df: pd.DataFrame) -> pd.DataFrame:
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
            proportion = lambda x: x.frequency_combined
                                   / x.frequency_combined.sum(),
            density = lambda x: x.proportion / x.interval_length
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
        .mark_rect(fill="#8FBC8F")
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
    
    return (histogram + labels)


def calibrate_beta_MM(observations):
    """Calibrate the parameters of the beta distribution on observations
    
    Source: https://statproofbook.github.io/P/beta-mome
    """
    sample_mean = np.mean(observations)
    sample_variance = np.var(observations, ddof=1) # unbiased sample variance
    
    a = sample_mean * (sample_mean * (1 - sample_mean) / sample_variance  - 1)
    b = a * (1 / sample_mean - 1)
    return a, b


def ramsey_population_certain(delta=0.01, eta=1, g_c=0.02, g_n=0.02, beta=1):
    discount_rate = (1 + delta) * (1 + g_n)**(-beta) * (1 + g_c)**eta - 1    
    return discount_rate


def population_size(t, n_0, g_n):
    return n_0 * (1 + g_n)**t


def beta_expectation(func, a, b):
    """    
    Compute the expected value of a function using a beta distribution.

    Return the expected value of `func(X)`, where random variable `X`
    follows a beta distribution of parameters `(a, b)`. 

    This is in theory equivalent to using scipy's `stats.beta.expect`
    method. However, for some values of parameters `a` and `b`, the
    integral cannot be computed using the `.expect` method.
    
    This function uses the `alg` weighting function available in the
    `scipy.integrate.quad` function. See the documentation of this
    function for more information. See also this StackOverflow question,
    which recommands this solution when scipy fails to calculate an
    integral: https://stackoverflow.com/q/44848189/12949296
    """
    beta_denominator = sp.special.beta(a, b)
    integrand = lambda x: func(x) / beta_denominator
    weight_parameters = (a - 1, b - 1)
    integration_tuple = sp.integrate.quad(
        integrand,
        0,
        1,
        weight="alg",
        wvar=weight_parameters
    )
    expectation = integration_tuple[0]
    return expectation


@np.vectorize
def ramsey_population_uncertain(t, a, b, n_0=10, g_n=0.02, delta=0.01, 
                                eta=1, g_c=0.02):
    population_value_future = lambda beta: population_size(t, n_0, g_n)**beta
    population_value_present = lambda beta: population_size(0, n_0, g_n)**beta
    
    population_expected_value_future = beta_expectation(
        population_value_future,
        a,
        b
    )
    population_expected_value_present = beta_expectation(
        population_value_present,
        a, 
        b
    )
    
    population_expected_value_ratio = (population_expected_value_future
                                       / population_expected_value_present)
    
    discount_rate = (
        (1 + delta) * (1 + g_c)**eta * population_expected_value_ratio**(-1/t)
        - 1
    )
    return discount_rate


# ======= Main code ========
# %% Read data
study_3c_path = "osfstorage-archive/Study 3c/PopEthics Study 3c.csv"
study_3c_df = pd.read_csv(study_3c_path)

# %% Process
processed_df = infer_beta_from_answers(study_3c_df)

# %% Histograms
beta_histogram_all = make_beta_histogram(processed_df, "all")
beta_histogram_90_70_50 = make_beta_histogram(processed_df, "90_70_50")
beta_histogram_k_m_b = make_beta_histogram(processed_df, "k_m_b")


# %% Calibrate
upper_bounds = processed_df.loc[:, 'all_upper'].dropna()
lower_bounds = processed_df.loc[:, 'all_lower'].dropna()

a_lower, b_lower = calibrate_beta_MM(lower_bounds)
a_upper, b_upper = calibrate_beta_MM(upper_bounds)

# %% Plot calibrated densities on histogram
beta_density_df = (
    pd.DataFrame({'x': np.linspace(0, 1, 1000)})
    .assign(
        beta_upper = lambda r: sp.stats.beta.pdf(r.x, a=a_upper, b=b_upper),
        beta_lower = lambda r: sp.stats.beta.pdf(r.x, a=a_lower, b=b_lower)
    )
)

beta_density_lower_line = f.line_chart(
    beta_density_df,
    x='x',
    y='beta_lower',
    color="#008B8B",
    x_title=""
)

beta_density_upper_line = f.line_chart(
    beta_density_df,
    x='x',
    y='beta_upper',
    color="#8018BF",
    strokeDash=[5,2],
    x_title=""
)

beta_calibration_plot = alt.layer(
    beta_histogram_all,
    beta_density_lower_line,
    beta_density_upper_line
)

beta_calibration_plot


# %%  Construct population discount rate dataframe
population_sdr_df = (
    pd.DataFrame()
    .assign(
        year=np.linspace(1, 500, 300),
        sdr_averagism=ramsey_population_certain(beta=0),
        sdr_totalism=ramsey_population_certain(beta=1),
        sdr_uniform=lambda x: ramsey_population_uncertain(t=x.year, a=1, b=1),
        sdr_beta_lower=lambda x: ramsey_population_uncertain(
            t=x.year,
            a=a_lower,
            b=b_lower
        ),
        sdr_beta_upper=lambda x: ramsey_population_uncertain(
            t=x.year,
            a=a_upper,
            b=b_upper
        )
    )
)


# %% Plot social discount rates with population growth
legend_dict = {
    "sdr_averagism": "Averagism",
    "sdr_beta_lower": "Beta uncertainty (most averagist)",
    "sdr_uniform": "Uniform uncertainty",
    "sdr_beta_upper": "Beta uncertainty (most totalist)",
    "sdr_totalism": "Totalism",
}

population_sdr_plot = (
    population_sdr_df
    .melt('year')
    .pipe(
        f.line_chart,
        x='year',
        y='value',
        multi=True,
        color='variable',
        strokeDash='variable',
        legend=alt.Legend(
            title=None,
            labelExpr=f.dict_to_labelExpr(legend_dict),
            orient="bottom"
        ),
        x_title="Year",
        y_title="Social discount rate",
        y_format="%"
    )
    .properties(width=550, height=300)
)

# %% Get Formsubmit data
# api_key = "..."
request_root = "https://formsubmit.co/api/get-submissions/"
request_text = request_root + api_key
request_result = requests.get(request_text)
print(request_result.text)

# %% Convert json to pandas dataframe
population_survey_df = pd.json_normalize(request_result.json(), "submissions")
population_survey_df.columns = (population_survey_df.columns
                                .str.replace(".+?\.", "", regex=True))
