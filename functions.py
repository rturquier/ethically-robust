#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions used for "Ethically robust discounting"
"""


import numpy as np
import pandas as pd
import altair as alt


def ramsey_beta_prime_gamma(tau, g, beta_prime, alpha, beta,
                            alpha_prime = 1, c_0=1,):
    """Return the social discount rate using equation (x).

    Arguments
    ---------
    tau         -- time period
    g           -- consumption growth rate
    alpha_prime -- first shape parameter of the beta prime distribution
                   only alpha_prime = 1 is supported
    beta_prime  -- second shape parameter of the beta prime distribution
    alpha       --  shape parameter of the gamma distribution
    beta        -- rate parameter of the gamma distribution
    c_0         -- consumption at tau = 0 (default 1)
    """
    discount_rate = (
        (1 / tau) * np.log(1 + tau / beta_prime)
        + (alpha / tau)
        * np.log(1 + tau * np.log(1 + g) / (beta + np.log(c_0)))
    )
    return discount_rate


def ramsey(delta, eta, g):
    return delta + eta * g


def exact_sdr_beta_prime_gamma(tau, g, beta_prime, alpha, beta,
                               alpha_prime = 1, c_0=1):
    """Return the exact social discount rate using equation (x).

    Arguments
    ---------
    tau         -- time period
    g           -- consumption growth rate
    alpha_prime -- first shape parameter of the beta prime distribution
                   only alpha_prime = 1 is supported
    beta_prime  -- second shape parameter of the beta prime distribution
    alpha       --  shape parameter of the gamma distribution
    beta        -- rate parameter of the gamma distribution
    c_0         -- consumption at tau = 0 (default 1)
    """
    discount_rate = (
        (1 + tau / beta_prime)**(1 / tau)
        * (
            1 + (tau * np.log(1 + g) / (beta + np.log(c_0)))
        )**(alpha / tau)
        - 1
    )
    return discount_rate


def sdr_to_sdf(sdr, tau):
    discount_factor = (1 / (1 + sdr))**tau
    return discount_factor


def exact_sdf_beta_prime_gamma(tau, g, beta_prime, alpha, beta,
                               alpha_prime = 1, c_0=1):
    """Return the exact social discount factor using equation (x).

    Arguments
    ---------
    tau         -- time period
    g           -- consumption growth rate
    alpha_prime -- first shape parameter of the beta prime distribution
                   only alpha_prime = 1 is supported
    beta_prime  -- second shape parameter of the beta prime distribution
    alpha       --  shape parameter of the gamma distribution
    beta        -- rate parameter of the gamma distribution
    c_0         -- consumption at tau = 0 (default 1)
    """
    sdr = exact_sdr_beta_prime_gamma(tau, g, beta_prime, alpha, beta,
                                     alpha_prime, c_0)
    discount_factor = sdr_to_sdf(sdr, tau)
    return discount_factor


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


def density_chart(df, x, freq, pdf, bin_step=1, x_format="~f",
                  bar_color="#8FBC8F", line_color="#008B8B", x_title="value",
                  y_title="density"):
    """Plot a histogram with a fitted probability density function."""

    bar_x     = alt.X(x, bin=alt.Bin(step=bin_step), axis=alt.Axis(title=""))
    bar_scale = 'datum.' + freq + ' / ' + str(bin_step)
    bar_y     = alt.Y('sum(freq_scaled):Q', axis=alt.Axis(title=""))

    line_x = alt.X(
        x,
        axis=alt.Axis(title=x_title,
                      format=x_format,
                      labelFlush=False,
                      zindex=1)
    )
    line_y = alt.Y(pdf,
                   axis=alt.Axis(title=y_title, zindex=1),
                   type="quantitative")


    base = alt.Chart(df)

    bar = (
        base.mark_bar()
        .transform_calculate(freq_scaled=bar_scale)
        .encode(x=bar_x, y=bar_y, color=alt.value(bar_color))
    )

    line = (
        base.mark_line(clip=True)
        .encode(x=line_x, y=line_y, color=alt.value(line_color))
    )
    
    complete_chart = (
        (bar + line)
        .properties(
            width=220,
            height=150
        )
    )

    return complete_chart


def dict_to_labelExpr(legend_dict):
    """Convert a dictonary to a labelExpr vega expression."""

    labelExpr = ""

    for index, variable_name in enumerate(legend_dict):
        labelExpr += f"datum.label == '{variable_name}' "
        labelExpr += f"? '{legend_dict[variable_name]}' : "

        if index + 1 == len(legend_dict):
            labelExpr += f"'unspecified label'"

    return labelExpr


def line_chart(df, x, y, x_title=False, y_title=False, x_format="~f",
               y_format="~f", y_scale=alt.Scale(), title="", subtitle="",
               color="#1f77b4", strokeDash=[1, 0], multi=False,
               legend=alt.Legend(), legend_order=None):
    """Plot a simple line chart."""

    if x_title == False:
        x_title = x
    if y_title == False:
        y_title = y

    line_chart_x_axis = alt.Axis(
        title=x_title,
        format=x_format,
        titlePadding=5
    )
    line_chart_y_axis = alt.Axis(
        title=y_title,
        format=y_format,
        titlePadding=10
    )

    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(x, axis=line_chart_x_axis),
            y=alt.Y(y, axis=line_chart_y_axis, scale=y_scale)
        )
        .properties(title={"text": title, "subtitle": subtitle})
    )

    if multi:
        chart = chart.encode(
            color=alt.Color(color, legend=legend, sort=legend_order),
            strokeDash=alt.StrokeDash(strokeDash, legend=legend,
                                      sort=legend_order)
        )
    else:
        chart = chart.encode(
            color=alt.value(color),
            strokeDash=alt.value(strokeDash)
        )

    return chart

