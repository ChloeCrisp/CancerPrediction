import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union

from IPython.display import display_html

def display_side_by_side(*args: list[pd.DataFrame]):
    """Function to display DataFrames side by side.

    Parameters
    ----------
    args
        List of dataframes to display side by side
    """
    html_str = ""
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace("table", 'table style="display:inline"'), raw=True)


def color_more_than_equal(
    row: pd.DataFrame, col: str, percentage: int, color: str = "#ccdd0a"
) -> list[str]:
    """Function to add colour to a pandas DataFrame if the value is more than or equal to a defined percentage.

    Parameters
    ----------
    row
        Row of a DataFrame
    col
        Name of a column to colour
    percentage
        Value representing percentage at which to add colour
    color
        String representing a hex colour code
    """
    if row.loc[col] >= percentage:
        return ["background-color: %s" % color] * len(row)

    return [""] * len(row)


def color_less_than_equal(
    row: pd.DataFrame, col: str, percentage: int, color: str = "#8dc73f"
) -> list[str]:
    """Function to add colour to a pandas DataFrame if the value is less than or equal to a defined percentage.

    Parameters
    ----------
    row
        Row of a DataFrame
    col
        Name of a column to colour
    percentage
        Value representing percentage at which to add colour
    color
        String representing a hex colour code
    """
    if row.loc[col] <= percentage:
        return ["background-color: %s" % color] * len(row)

    return [""] * len(row)

def response_over_time(
    data: pd.DataFrame,
    plot_dict: dict[str, Union[str, tuple[str]]],
    rate_plot: bool = False,
):
    """Plots response frequency or rate grouped by given time column

    Parameters
    ----------
    data
        df containing two data columns to plot ([response, group_by])
    plot_dict
        dictionary containing details for making plot, including:
         "response column" : str
            response column string
         "group by" : str
            column string of time values by which to group data
         "response labels" : tuple
            tuple containing response label strings
    rate_plot
        False : plots frequency (number of bad and good risk responses)
        True : plots rate (proportion of bad risk responses)
    """

    response = plot_dict["response column"]
    groups = plot_dict["group by"]

    df_freq = (
        data.groupby([groups, response])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={0: "good", 1: "bad"})
    )

    if pd.api.types.is_datetime64_any_dtype(df_freq[groups]):
        group_labels = df_freq[groups].dt.strftime("%Y-%m")
    else:
        group_labels = df_freq[groups]

    num_months = df_freq.shape[0]
    ind = list(range(num_months))
    bar_width = 0.35

    figsize = (15, 5)

    if rate_plot:
        df_freq["resp_rate"] = df_freq["bad"] / df_freq[["bad", "good"]].sum(axis=1)
        ax = df_freq[[groups, "resp_rate"]].plot(
            x=plot_dict["group by"],
            y="resp_rate",
            legend=False,
            figsize=figsize,
            marker="o",
            title="Response rate over time",
        )
        ax.tick_params(axis="x", labelrotation=90)
        plt.tight_layout()
        plt.show()

    else:
        fig = plt.figure(figsize=figsize)
        fig.add_subplot(111)
        plt.bar(
            [x - 0 for x in ind],
            df_freq["bad"],
            bar_width,
            label=plot_dict["response labels"][1],
        )
        plt.bar(
            [x + bar_width for x in ind],
            df_freq["good"],
            bar_width,
            label=plot_dict["response labels"][0],
        )
        plt.xticks(
            [x + 0.5 * bar_width for x in ind], group_labels, rotation=90
        )
        plt.legend()
        plt.title("Response frequencies over time")
        plt.tight_layout()
        plt.show()

def distribution_plots(
    df: pd.DataFrame,
    categorical_plot: bool = False,
    ncols: int = 3,
    width: int = 6,
    height: int = 5,
):
    """Plots distributions of feature values for numerical or categorical values

    Parameters
    ----------
    df
        df of features to plot
    categorical_plot
        If true plot categorical distributions, otherwise if false plot numerical distributions
    ncols
        number of columns of plots
    width
        width of each plot
    height
        height of each plot
    """

    fig_h = height * (df.shape[1] // ncols + (df.shape[0] % ncols > 0))
    fig, ax = plt.subplots(
        nrows=((df.shape[1] + ncols) // ncols),
        ncols=ncols,
        figsize=(width * ncols, fig_h),
    )
    colors = "bgrcmyk" * ((df.shape[1] // 7) + 1)

    for idx, col in enumerate(df.columns):
        x, y = idx // ncols, idx % ncols

        if categorical_plot:
            df[col].value_counts().plot(
                ax=ax[x, y],
                kind="bar",
                legend=False,
                color=colors[idx],
                rot=50,
                fontsize=15,
            )
            ax[x, y].set_title(col, fontsize="large")

        else:
            df[col].plot(
                ax=ax[x, y], kind="hist", legend=False, color=colors[idx], title=col
            )

    plt.tight_layout()

def correlation_plot(data: pd.DataFrame):
    """Plots correlation heatmap between input features data

    Parameters
    ----------
    data
        dataframe of columns to plot correlations over
    """

    # calculate the correlation matrix
    corr = data.corr()

    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the heatmap
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, ax=ax)