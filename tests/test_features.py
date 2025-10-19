"""
Unit tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from ml_framework import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    dates = pd.date_range('2024-01-01 09:00', periods=1000, freq='1min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': 38000 + np.cumsum(np.random.randn(1000) * 10),
        'high': 38000 + np.cumsum(np.random.randn(1000) * 10) + 50,
        'low': 38000 + np.cumsum(np.random.randn(1000) * 10) - 50,
        'close': 38000 + np.cumsum(np.random.randn(1000) * 10),
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    return df


def test_feature_engineer_initialization():
    """Test FeatureEngineer initialization."""
    engineer = FeatureEngineer()
    assert engineer is not None


def test_create_all_features(sample_data):
    """Test creating all features."""
    engineer = FeatureEngineer()
    result = engineer.create_all_features(sample_data)
    
    # Check output is DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check original columns preserved
    assert 'open' in result.columns
    assert 'high' in result.columns
    assert 'low' in result.columns
    assert 'close' in result.columns
    assert 'volume' in result.columns
    
    # Check feature columns added
    assert 'returns' in result.columns
    assert 'rsi_14' in result.columns
    assert 'macd' in result.columns
    assert 'atr_14' in result.columns
    
    # Check no infinite values in clean data
    clean_df = result.dropna()
    assert not np.isinf(clean_df.select_dtypes(include=[np.number]).values).any()


def test_feature_count(sample_data):
    """Test correct number of features created."""
    engineer = FeatureEngineer()
    result = engineer.create_all_features(sample_data)
    
    # Should have original 6 columns + ~63 features
    assert len(result.columns) >= 60


def test_nan_handling(sample_data):
    """Test NaN handling in features."""
    engineer = FeatureEngineer()
    result = engineer.create_all_features(sample_data)
    
    # Initial rows will have NaN due to indicators
    assert result.isna().any().any()
    
    # After dropna, should be clean
    clean_df = result.dropna()
    assert len(clean_df) > 0
    assert not clean_df.isna().any().any()


def test_feature_ranges(sample_data):
    """Test feature value ranges are reasonable."""
    engineer = FeatureEngineer()
    result = engineer.create_all_features(sample_data)
    clean_df = result.dropna()
    
    # RSI should be between 0 and 100
    assert clean_df['rsi_14'].min() >= 0
    assert clean_df['rsi_14'].max() <= 100
    
    # ATR should be positive
    assert clean_df['atr_14'].min() >= 0


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    engineer = FeatureEngineer()
    empty_df = pd.DataFrame()
    
    with pytest.raises((KeyError, ValueError)):
        engineer.create_all_features(empty_df)


def test_missing_columns():
    """Test handling of missing required columns."""
    engineer = FeatureEngineer()
    incomplete_df = pd.DataFrame({
        'open': [100, 101, 102],
        'close': [101, 102, 103]
    })
    
    with pytest.raises(KeyError):
        engineer.create_all_features(incomplete_df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
