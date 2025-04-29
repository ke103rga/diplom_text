import pandas as pd
import numpy as np
from typing import List


class TimeSeriesAnalysisResult:
    def __init__(self):
        self.smoothed_series = {}
        self.autocorrelation = None
        self.trend_line = None


class TimeSeriesAnalyzer:
    
    @staticmethod
    def smooth_time_series(data: pd.Series, window_size: int) -> pd.Series:
        """
        Applies moving average to smooth a time series.
        
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
    def autocorrelation(x: np.ndarray, lags: int) -> np.ndarray:
        """
        Computes the autocorrelation of a time series. To detect periods of seasonality in the time series.
        
        Args:
            x (np.ndarray): The values of the time series.
            lags (int): The number of lags for autocorrelation.
        
        Returns:
            np.ndarray: The autocorrelation values for each lag.
        """
        n = len(x)
        mean = np.mean(x)
        c0 = np.var(x) * n
        autocorr = np.correlate(x - mean, x - mean, mode='full')[-n:]
        result = autocorr / c0
        
        return result[:lags + 1]

    @staticmethod
    def trend_line(data: pd.Series) -> pd.Series:
        """
        Calculates the trend line for a time series. Calculates by fittinf OLS regression to the data.
        and computing the linear regression func by the original x values.
        
        Args:
            data (pd.Series): The time series.
        
        Returns:
            pd.Series: The values of the trend line.
        """
        x_range = np.arange(len(data))
        trend_line_coeficients = np.polyfit(x_range, data, 1)  # 1 for linear regression
        trend_line_func = np.poly1d(trend_line_coeficients)
        return trend_line_func(x_range)
    
    @staticmethod
    def quantitative_trend_analysis(data: pd.Series) -> pd.DataFrame:
        """
        Performs quantitative trend analysis of a time series and returns a DataFrame with the scores.

        Args:
            data (pd.Series): The time series for trend analysis.
        
        Returns:
            pd.DataFrame: A DataFrame containing method names and their trend evaluations.
        """
        results = []

        # 1. Linear regression slope
        x = np.arange(len(data))
        slope, intercept = np.polyfit(x, data, 1)  # Получаем наклон и пересечение
        results.append(['Linear regression slope', slope])
        
        # 2. Correaltion with increasing time series
        increasing_series = np.arange(1, len(data) + 1)  
        correlation = np.corrcoef(data, increasing_series)[0, 1]  
        results.append(['Corr with increasing ts', correlation])
        
        # Result DataFrame
        trend_df = pd.DataFrame(results, columns=['method', 'result'])
        return trend_df
    
    @staticmethod
    def analyze(data: pd.Series, window_sizes: List[int] = None, lags: int = None, 
                trend_line: bool = False) -> TimeSeriesAnalysisResult:
        """
        Analyzes the time series by applying smoothing, calculating autocorrelation, and optionally evaluating the trend line.

        Args:
            data (pd.Series): The time series to analyze.
            window_sizes (List[int], optional): List of window sizes for smoothing. Defaults to None.
            lags (int, optional): Number of lags for autocorrelation calculation. Defaults to None.
            trend_line (bool, optional): Whether to calculate the trend line. Defaults to False.

        Returns:
            TimeSeriesAnalysisResult: An object containing the results of the analysis.
        """
        result = TimeSeriesAnalysisResult()
        
        if window_sizes is not None:
            for window_size in window_sizes:
                smoothed = TimeSeriesAnalyzer.smooth_time_series(data, window_size)
                result.smoothed_series[window_size] = smoothed

        if lags is not None:
            result.autocorrelation = TimeSeriesAnalyzer.autocorrelation(data.values, lags)

        if trend_line:
            result.trend_line = TimeSeriesAnalyzer.trend_line(data)
        
        return result
    