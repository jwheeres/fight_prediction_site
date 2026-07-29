import math

import pandas as pd
import pytest

from trading.indicators import atr, rsi, sma


def test_sma_hand_computed():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(series, window=3)
    expected = [math.nan, math.nan, 2.0, 3.0, 4.0]
    for got, want in zip(result, expected):
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(want)


def test_rsi_hand_computed():
    close = pd.Series([10.0, 11.0, 10.0, 12.0, 11.0, 13.0])
    result = rsi(close, period=2)
    expected = [math.nan, math.nan, 50.0, 66.6666667, 66.6666667, 66.6666667]
    for got, want in zip(result, expected):
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(want, rel=1e-6)


def test_rsi_flat_price_is_50():
    close = pd.Series([10.0, 10.0, 10.0, 10.0])
    result = rsi(close, period=2)
    assert result.iloc[2] == pytest.approx(50.0)
    assert result.iloc[3] == pytest.approx(50.0)


def test_atr_hand_computed():
    high = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0])
    low = pd.Series([9.0, 9.0, 10.0, 9.0, 11.0])
    close = pd.Series([9.5, 10.5, 11.0, 10.0, 12.0])
    result = atr(high, low, close, period=2)
    expected = [math.nan, 1.5, 2.0, 2.0, 2.5]
    for got, want in zip(result, expected):
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(want)
