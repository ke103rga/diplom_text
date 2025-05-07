import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from typing import Optional, List, Union, Literal, Callable, Tuple
import matplotlib.pyplot as plt

from metrics.time_series_analyzer import TimeSeriesAnalyzer


class DinamicMetricPlotter:

    @staticmethod
    def plot(data: pd.DataFrame, dt_col: str, value_col: str,
             aggfunc: Union[str, Callable] = 'sum',
             hue_cols: Optional[List[str]] = None,
             smooth: int = 0,
             mode: Literal['bar', 'line', 'auto'] = 'auto',
             engine: str = 'plotly',
             figsize: Tuple[int, int] = (12, 6)):
        """
        Plots the data using seaborn or plotly. 

        Args:
            data (pd.DataFrame): The data to plot.
            dt_col (str): The column name for the date or time variable.
            value_col (str): The column name for the values to be plotted.
            aggfunc (Union[str, Callable], optional): The aggregation function to apply. Defaults to 'sum'.
            hue_cols (Optional[List[str]], optional): The columns for grouping the data. Defaults to None.
            smooth (int, optional): The window size for smoothing. Defaults to 0 (no smoothing).
            mode (Literal['bar', 'line', 'auto'], optional): The mode for the plot. Defaults to 'auto'.
            engine (str, optional): The plotting engine. Defaults to 'plotly'.
            figsize (Tuple[int, int], optional): The size of the figure. Defaults to (12, 6).

        Raises:
            ValueError: If the engine is not one of the allowed values.
        """
        if engine not in ['seaborn', 'plotly']:
            raise ValueError("Engine must be either 'seaborn' or 'plotly'")
        # Prepare plot data by splitting by hue columns and applying aggregation
        plot_data = DinamicMetricPlotter._prepare_plot_data(data, dt_col, value_col, hue_cols, aggfunc)
        # Prepare subsets (separate df for each hue combination). Smoothing is applied for each subset.
        subsets, labels = DinamicMetricPlotter._prepare_subsets(plot_data, value_col, hue_cols, smooth)
        # Define plotting mode (line or bar). 
        mode = DinamicMetricPlotter._define_mode(plot_data, dt_col, mode)

        if engine == 'seaborn':
            with sns.axes_style('darkgrid'):
                fig, ax = plt.subplots(figsize=figsize)
                for subset, label in zip(subsets, labels):
                    if mode == 'line':
                        sns.lineplot(data=subset, x=dt_col, y=value_col, label=label, ax=ax)
                    else:
                        sns.barplot(data=subset, x=dt_col, y=value_col, label=label, ax=ax)
                plt.tight_layout()

        else:
            fig = go.Figure()

            for subset, label in zip(subsets, labels):
                if mode == 'line':
                    fig.add_trace(go.Scatter(
                        x=subset[dt_col],
                        y=subset[value_col],
                        mode='lines',
                        name=label
                    ))
                else:
                    fig.add_trace(go.Bar(
                        x=subset[dt_col],
                        y=subset[value_col],
                        name=label,
                        hoverinfo='y+x',
                        showlegend=True
                    ))

            fig.update_layout(
                xaxis_title=dt_col,
                yaxis_title=value_col,
                barmode='group' if mode == 'bar' else 'overlay',  # Установка режима для столбчатых графиков
            )
            fig.show()

    @staticmethod
    def plot_analysis(data: pd.DataFrame, dt_col: str, value_col: str,
                      aggfunc: Union[str, Callable] = 'sum',
                      window_sizes: Optional[List[int]] = None,
                      lags: Optional[int] = None,
                      trend_line: bool = False,
                      figsize: Optional[Tuple[int, int]] = None):
        """
        Analyzes the time series data and plots the original values, smoothed values, and optionally the trend line
        and autocorrelation.

        Args:
            data (pd.DataFrame): The time series data to analyze.
            dt_col (str): The column name for the date or time variable.
            value_col (str): The column name for the values to be plotted.
            aggfunc (Union[str, Callable], optional): The aggregation function to apply. Defaults to 'sum'.
            window_sizes (Optional[List[int]], optional): List of window sizes for smoothing. Defaults to None.
            lags (Optional[int], optional): Number of lags for autocorrelation calculation. Defaults to None.
            trend_line (bool, optional): Whether to calculate the trend line. Defaults to False.
            figsize (Optional[Tuple[int, int]], optional): The size of the figure. Defaults to None.

        Returns:
            None
        """
        # Prepare plot data by aggregatin all values by dt col
        # After applying _prepare_plot_data there are only one value_col value for each dt_col value
        plot_data = DinamicMetricPlotter._prepare_plot_data(data=data, dt_col=dt_col, value_col=value_col,
                                                            aggfunc=aggfunc)

        # Get the analysis results
        analysis_result = TimeSeriesAnalyzer.analyze(
            plot_data[value_col],
            window_sizes=window_sizes,
            lags=lags,
            trend_line=trend_line
        )

        # Determine the number of plots and adjust the figure size accordingly
        num_plots = 1 if lags is None else 2
        if figsize is None:
            figsize = (12, 4 * num_plots)

        with sns.axes_style('darkgrid'):
            gridspec_kw = {'height_ratios': [4, 1]} if num_plots == 2 else {}
            fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=False, gridspec_kw=gridspec_kw)

            # Plot the original values 
            ax = axes if lags is None else axes[0]
            sns.lineplot(data=plot_data, x=dt_col, y=value_col, ax=ax, label='Original Data')

            # ad smoothed values for the same axis
            for window_size, smoothed in analysis_result.smoothed_series.items():
                sns.lineplot(x=plot_data[dt_col], y=smoothed, ax=ax, label=f'Smoothed (window={window_size})')

            # Add trend line for the same axis
            if trend_line and analysis_result.trend_line is not None:
                sns.lineplot(x=plot_data[dt_col], y=analysis_result.trend_line, ax=ax, label='Trend Line', color='red')

            ax.set_title('Time Series Analysis')
            ax.legend()

            # Add autocorrelation plot
            if lags is not None:
                ax2 = axes[1]
                autocorr_data = analysis_result.autocorrelation
                # Create vertical lines for each data point and markers on the plot
                ax2.scatter(np.arange(len(autocorr_data)), autocorr_data, color='b', marker='o')
                for i, value in enumerate(autocorr_data):
                    ax2.vlines(x=i, ymin=0, ymax=value, color='b')  # Vertical lines

                ax2.set_title('Autocorrelation')
                ax2.set_ylabel('Autocorrelation')
                ax2.set_xlabel('Lags')
                ax2.axhline(0, color='b', linewidth=0.5, alpha=0.5)

                plt.tight_layout()
                plt.show()

    @staticmethod
    def smooth_time_series(data: pd.Series, window_size: int) -> pd.Series:
        """
        Applies moving average to smooth the time series.
        
        Args:
            data (pd.Series): The time series to be smoothed.
            window_size (int): The size of the window for the moving average.
        
        Returns:
            pd.Series: The smoothed time series.
        """
        if window_size < 1:
            return data
        return data.rolling(window=window_size, center=False).mean()

    @staticmethod
    def _prepare_plot_data(data: pd.DataFrame, dt_col: str, value_col: str, hue_cols: Optional[List[str]] = None,
                           aggfunc: Union[str, Callable] = 'sum') -> pd.DataFrame:
        """
        Prepares the data for plotting by grouping and aggregating based on the specified columns.

        Args:
            data (pd.DataFrame): The original data to prepare.
            dt_col (str): The column name for the date or time variable.
            value_col (str): The column name for the values to be plotted.
            hue_cols (Optional[List[str]], optional): The columns for grouping the data. Defaults to None.
            aggfunc (Union[str, Callable], optional): The aggregation function to apply. Defaults to 'sum'.
        
        Returns:
            pd.DataFrame: The prepared data for plotting.
        """
        if isinstance(hue_cols, str):
            hue_cols = [hue_cols]
        elif hue_cols is None or len(hue_cols) == 0:
            hue_cols = []

            # Agg data. If there are only one value in each group then return value else return aggregation result
        data = data.groupby([dt_col] + hue_cols).agg(**{
            value_col: (value_col, aggfunc)
        }).reset_index()
        return data

    @staticmethod
    def _prepare_subsets(data: pd.DataFrame, value_col: str, hue_cols: Optional[List[str]] = None,
                         smooth: int = 0) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Prepares subsets of the data for plotting based on hue columns.

        Args:
            data (pd.DataFrame): The data to prepare subsets from.
            value_col (str): The column name for the values to be plotted.
            hue_cols (Optional[List[str]], optional): The columns for grouping the data. Defaults to None.
            smooth (int, optional): The window size for smoothing. Defaults to 0.
        
        Returns:
            Tuple[List[pd.DataFrame], List[str]]: A tuple containing a list of subsets and their corresponding labels.
        """
        if hue_cols is None or len(hue_cols) == 0:
            data.loc[:, value_col] = DinamicMetricPlotter.smooth_time_series(data[value_col], smooth)
            return [data], ['']

        subsets = []
        labels = []
        if len(hue_cols) == 1:
            unique_hues = data[hue_cols[0]].unique()
            for hue in unique_hues:
                subset = data[data[hue_cols[0]] == hue]
                subset.loc[:, value_col] = DinamicMetricPlotter.smooth_time_series(subset[value_col], smooth)
                subsets.append(subset)
                labels.append(hue)
        else:
            unique_combinations = data[hue_cols].drop_duplicates()
            for _, combo in unique_combinations.iterrows():
                combo_label = ' & '.join(combo.apply(lambda x: f"{combo.index[combo == x][0]}={x}"))
                subset = data[(data[hue_cols] == combo).all(axis=1)]
                subset.loc[:, value_col] = DinamicMetricPlotter.smooth_time_series(subset[value_col], smooth)
                subsets.append(subset)
                labels.append(combo_label)
        return subsets, labels

    @staticmethod
    def _define_mode(data: pd.DataFrame, dt_col: str, mode: Literal['bar', 'line', 'auto']) -> Literal['bar', 'line']:
        return 'line'
        # if mode == 'auto':
        #     unique_x_values = data[dt_col].nunique()
        #     return 'line' if unique_x_values > 20 else 'bar'
        # return mode
