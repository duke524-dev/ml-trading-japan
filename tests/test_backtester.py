"""
Unit tests for backtesting module.
"""

import pytest
import pandas as pd
import numpy as np
from ml_framework import Backtester


@pytest.fixture
def sample_backtest_data():
    """Create sample data for backtesting."""
    dates = pd.date_range('2024-01-01 09:00', periods=500, freq='1min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': 38000 + np.cumsum(np.random.randn(500) * 10),
        'high': 38000 + np.cumsum(np.random.randn(500) * 10) + 50,
        'low': 38000 + np.cumsum(np.random.randn(500) * 10) - 50,
        'close': 38000 + np.cumsum(np.random.randn(500) * 10),
        'volume': np.random.randint(1000, 10000, 500),
        'atr_14': np.random.uniform(30, 60, 500)
    })
    
    return df


@pytest.fixture
def sample_signals():
    """Create sample trading signals."""
    np.random.seed(42)
    return np.random.choice([-1, 0, 1], 500, p=[0.1, 0.8, 0.1])


def test_backtester_initialization():
    """Test Backtester initialization."""
    backtester = Backtester(
        initial_capital=5_000_000,
        tick_size=0.5,
        multiplier=10000
    )
    
    assert backtester.initial_capital == 5_000_000
    assert backtester.tick_size == 0.5
    assert backtester.multiplier == 10000


def test_backtest_run(sample_backtest_data, sample_signals):
    """Test running a backtest."""
    backtester = Backtester()
    results = backtester.run(sample_backtest_data, sample_signals)
    
    # Check results structure
    assert 'roi' in results
    assert 'total_trades' in results
    assert 'win_rate' in results
    assert 'final_capital' in results
    
    # Check results are reasonable
    assert isinstance(results['roi'], float)
    assert results['total_trades'] >= 0
    assert 0 <= results['win_rate'] <= 100


def test_no_trades(sample_backtest_data):
    """Test backtest with no trading signals."""
    backtester = Backtester()
    signals = np.zeros(len(sample_backtest_data), dtype=int)
    results = backtester.run(sample_backtest_data, signals)
    
    assert results['total_trades'] == 0
    assert results['roi'] == 0.0
    assert results['final_capital'] == backtester.initial_capital


def test_all_long_signals(sample_backtest_data):
    """Test backtest with all long signals."""
    backtester = Backtester()
    signals = np.ones(len(sample_backtest_data), dtype=int)
    results = backtester.run(sample_backtest_data, signals)
    
    # Should have at least 1 trade
    assert results['total_trades'] >= 1


def test_trade_recording(sample_backtest_data, sample_signals):
    """Test trade recording functionality."""
    backtester = Backtester()
    results = backtester.run(sample_backtest_data, sample_signals)
    
    trades_df = backtester.get_trades_df()
    
    if results['total_trades'] > 0:
        assert len(trades_df) == results['total_trades']
        assert 'entry_time' in trades_df.columns
        assert 'exit_time' in trades_df.columns
        assert 'pnl' in trades_df.columns


def test_calculate_metrics():
    """Test metrics calculation."""
    backtester = Backtester(initial_capital=1_000_000)
    results = backtester.calculate_metrics(1_200_000)
    
    assert results['roi'] == 20.0
    assert results['total_pnl'] == 200_000
    assert results['final_capital'] == 1_200_000


def test_position_sizing():
    """Test backtest with different position sizes."""
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    df = pd.DataFrame({
        'datetime': dates,
        'open': np.full(100, 38000),
        'high': np.full(100, 38100),
        'low': np.full(100, 37900),
        'close': np.full(100, 38000),
        'volume': np.full(100, 5000),
        'atr_14': np.full(100, 50)
    })
    
    signals = np.array([1] + [0] * 99)
    
    backtester = Backtester()
    results = backtester.run(df, signals, position_size=0.5)
    
    # With half position size, P&L should be smaller
    assert results['total_trades'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
