from pandas import DataFrame, options, isnull
import numpy as np


def modify_col(column):
    new_column = column.sort_values(ignore_index=True)
    new_column.dropna(axis=0, inplace=True)
    return (new_column)


def ft_count(column):
    total = 0
    for row in column:
        total += 1
    return total


def ft_mean(column, total):
    sumr = 0
    for row in column:
        sumr += row
    return (sumr / total)


def interpolation(column, total, quantile):
    """Quantiles calculated with linear interpolation :
    y = ya + (yb - ya) * ((x - xa) / (xb - xa))
    x is the quartile, a and b are its upper and lower direct values."""

    x = (total - 1) * quantile
    yb = column[int(x)]
    xb = int(x)
    ya = column[int(x) + 1]
    xa = int(x) + 1
    to_multiply = (x - xa) / (xb - xa)
    y = ya + ((yb - ya) * to_multiply)
    return (y)


def ft_quartiles(column, total, desc_df, column_name):
    """A quartile is value, that a certain percentage will stand above.
    i.e : in a range from 1 to 10, second quartile (50%) would be 5.5,
    so half of value would stand below and half above"""

    desc_df.loc['25%', column_name] = interpolation(column, total, 0.25)
    desc_df.loc['50%', column_name] = interpolation(column, total, 0.5)
    desc_df.loc['75%', column_name] = interpolation(column, total, 0.75)


def ft_var(column, total, mean):
    """Variance explains how spreads out datas in a set of number.
    The largest, the more it spreads."""

    sum = 0
    for row in column:
        sum += (row - mean) ** 2
    return (sum / (total - 1))


def ft_std(variance):
    """Standard deviation is the square root of variance.
    It brings back to the same units as the original data, 
    making interepretation easier."""

    return (variance ** 0.5)


def calculate_for_col(column, desc_df, column_name):
    total = ft_count(column)
    mean = ft_mean(column, total)

    desc_df.loc['Count', column_name] = total
    desc_df.loc['Mean', column_name] = mean
    desc_df.loc['Min', column_name] = column[0]
    desc_df.loc['Max', column_name] = column[total - 1]
    ft_quartiles(column, total, desc_df, column_name)
    desc_df.loc['Var', column_name] = ft_var(column, total, mean)
    desc_df.loc['Std', column_name] = ft_std(desc_df[column_name]['Var'])


def describe(data: DataFrame):
    """Homemade describe function"""

    options.display.float_format = '{:.6f}'.format

    # select all numerics datas
    features = data.select_dtypes(include=np.number)
    if features.empty:
        return (DataFrame())
    columns = features.columns
    desc_df = DataFrame(columns=columns,
                        index=["Count", "Mean", "Std", "Min", "25%",
                               "50%", "75%", "Max"],
                        dtype=float
                        )
    for col in columns:
        sorted_column = modify_col(features[col])
        calculate_for_col(sorted_column, desc_df, col)

    return (desc_df)