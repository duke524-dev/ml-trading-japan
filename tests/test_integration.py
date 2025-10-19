"""
Integration tests - End-to-end workflow.
"""

import pytest
import pandas as pd
import numpy as np
from ml_framework import (
    FeatureEngineer,
    MLSignalPredictor,
    Backtester,
    WalkForwardOptimizer,
    get_market_config
)


@pytest.fixture
def sample_trading_data():
    """Create realistic trading data."""
    dates = pd.date_range('2024-01-01 09:00', periods=2000, freq='1min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': 38000 + np.cumsum(np.random.randn(2000) * 10),
        'high': 38000 + np.cumsum(np.random.randn(2000) * 10) + 50,
        'low': 38000 + np.cumsum(np.random.randn(2000) * 10) - 50,
        'close': 38000 + np.cumsum(np.random.randn(2000) * 10),
        'volume': np.random.randint(1000, 10000, 2000)
    })
    
    return df


def test_full_pipeline(sample_trading_data):
    """Test complete trading pipeline."""
    df = sample_trading_data
    
    # 1. Feature engineering
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    df = df.dropna()
    
    assert len(df) > 100
    
    # 2. Train/test split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # 3. Generate labels
    lookahead = 20
    train_df['future_high'] = train_df['high'].rolling(lookahead, min_periods=1).max().shift(-lookahead)
    train_df['future_low'] = train_df['low'].rolling(lookahead, min_periods=1).min().shift(-lookahead)
    
    upside = (train_df['future_high'] - train_df['close']) / train_df['close']
    downside = (train_df['close'] - train_df['future_low']) / train_df['close']
    
    train_df['label'] = 0
    train_df.loc[upside >= 0.05, 'label'] = 1
    train_df.loc[downside >= 0.05, 'label'] = -1
    
    train_df = train_df.iloc[:-lookahead]
    
    # 4. Train model
    exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'label', 
                    'future_high', 'future_low', 'upside', 'downside']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train_df['label'].values
    
    model = MLSignalPredictor()
    model.train(X_train, y_train)
    
    # 5. Generate signals
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    signals = model.predict(X_test, confidence_threshold=0.20)
    
    assert len(signals) == len(test_df)
    
    # 6. Backtest
    backtester = Backtester()
    results = backtester.run(test_df, signals)
    
    assert 'roi' in results
    assert 'total_trades' in results
    assert results['total_trades'] >= 0


def test_market_config_integration():
    """Test market configuration integration."""
    config = get_market_config('topix')
    
    backtester = Backtester(
        initial_capital=5_000_000,
        tick_size=config.tick_size,
        multiplier=config.multiplier
    )
    
    assert backtester.tick_size == 0.5
    assert backtester.multiplier == 10000


def test_optimizer_integration(sample_trading_data):
    """Test optimizer with small grid."""
    df = sample_trading_data
    
    # Create features
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    df = df.dropna()
    
    # Small parameter grid
    param_grid = {
        'profit_threshold': [0.05, 0.08],
        'lookahead_bars': [15, 20],
        'ml_confidence': [0.15, 0.20],
        'tp_multiplier': [2.0, 3.0],
        'sl_multiplier': [1.0],
        'trail_multiplier': [1.2]
    }
    
    optimizer = WalkForwardOptimizer(n_splits=2, min_trades=5)
    
    # Should complete without errors
    best = optimizer.optimize(df, param_grid, verbose=False)
    
    assert 'params' in best
    assert 'avg_roi' in best
    assert isinstance(best['avg_roi'], float)


def test_feature_importance_integration(sample_trading_data):
    """Test feature importance extraction."""
    df = sample_trading_data
    
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    df = df.dropna()
    
    # Generate simple labels
    df['label'] = np.where(df['close'].pct_change().shift(-1) > 0, 1, -1)
    
    exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = df['label'].values
    
    model = MLSignalPredictor()
    model.train(X, y)
    
    importance_df = model.get_feature_importance(feature_cols)
    
    assert len(importance_df) == len(feature_cols)
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
