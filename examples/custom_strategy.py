"""
Custom Strategy Example

Template for creating your own trading strategy.

Customize:
1. Feature engineering (add custom indicators)
2. Label generation (different profit thresholds)
3. ML model parameters (XGBoost tuning)
4. Backtest parameters (TP/SL/Trail levels)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from ml_framework import (
    FeatureEngineer,
    MLSignalPredictor,
    Backtester,
    MarketConfig
)


class CustomFeatureEngineer(FeatureEngineer):
    """
    Extend FeatureEngineer with your custom indicators.
    """
    
    def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add your custom technical indicators here.
        
        Example: Add custom momentum indicator
        """
        # Custom Momentum Score (example)
        df['custom_momentum'] = (
            df['close'].pct_change(5) * 0.3 +
            df['close'].pct_change(10) * 0.3 +
            df['close'].pct_change(20) * 0.4
        )
        
        # Custom Volatility-Adjusted Returns
        df['vol_adj_returns'] = df['close'].pct_change() / df['atr_14']
        
        # Custom Order Flow Proxy
        df['buy_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['sell_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-10)
        
        return df


def custom_label_generation(df: pd.DataFrame,
                            profit_threshold: float = 0.10,
                            loss_threshold: float = 0.05,
                            lookahead: int = 30) -> pd.DataFrame:
    """
    Custom label generation with asymmetric thresholds.
    
    Args:
        df: DataFrame with OHLCV
        profit_threshold: Profit target (e.g., 0.10 = 10%)
        loss_threshold: Loss threshold (e.g., 0.05 = 5%)
        lookahead: Bars to look ahead
    
    Returns:
        DataFrame with 'label' column
    """
    df = df.copy()
    
    # Calculate future max/min
    df['future_high'] = df['high'].rolling(window=lookahead, min_periods=1).max().shift(-lookahead)
    df['future_low'] = df['low'].rolling(window=lookahead, min_periods=1).min().shift(-lookahead)
    
    # Calculate potential returns
    upside = (df['future_high'] - df['close']) / df['close']
    downside = (df['close'] - df['future_low']) / df['close']
    
    # Asymmetric labeling (higher bar for Long, lower for Short)
    df['label'] = 0
    df.loc[upside >= profit_threshold, 'label'] = 1  # Long
    df.loc[downside >= loss_threshold, 'label'] = -1  # Short
    
    return df


def main():
    print("\n" + "="*70)
    print("Custom Trading Strategy Example")
    print("="*70)
    
    # 1. Load your data
    print("\n[Step 1] Loading data...")
    data_path = Path(__file__).parent.parent / "data" / "TOPIX_18months_clean.csv"
    
    if not data_path.exists():
        print(f"⚠️  Data file not found: {data_path}")
        print("   Using sample data...")
        # Generate sample data
        dates = pd.date_range('2024-01-01 09:00', periods=10000, freq='1min')
        df = pd.DataFrame({
            'datetime': dates,
            'open': 2400 + np.cumsum(np.random.randn(10000) * 2),
            'high': 2400 + np.cumsum(np.random.randn(10000) * 2) + 5,
            'low': 2400 + np.cumsum(np.random.randn(10000) * 2) - 5,
            'close': 2400 + np.cumsum(np.random.randn(10000) * 2),
            'volume': np.random.randint(1000, 50000, 10000)
        })
    else:
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"✅ Loaded {len(df):,} rows")
    
    # 2. Create features (including custom ones)
    print("\n[Step 2] Creating features (base + custom)...")
    engineer = CustomFeatureEngineer()
    df = engineer.create_all_features(df)
    df = engineer.add_custom_features(df)
    
    df = df.dropna()
    print(f"✅ Features created: {len(df):,} rows")
    
    # 3. Custom label generation
    print("\n[Step 3] Generating custom labels...")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Use custom labeling function
    train_df = custom_label_generation(
        train_df,
        profit_threshold=0.12,  # Higher profit target
        loss_threshold=0.06,     # Lower loss threshold (easier to short)
        lookahead=25
    )
    
    train_df = train_df.iloc[:-25]  # Remove last lookahead rows
    
    label_dist = train_df['label'].value_counts()
    print(f"✅ Labels generated:")
    print(f"   Long: {label_dist.get(1, 0):,}, Neutral: {label_dist.get(0, 0):,}, Short: {label_dist.get(-1, 0):,}")
    
    # 4. Train model with custom parameters
    print("\n[Step 4] Training XGBoost with custom parameters...")
    
    exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'label']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train_df['label'].values
    
    # Custom XGBoost parameters
    model = MLSignalPredictor(
        n_estimators=300,      # More trees
        max_depth=8,           # Deeper trees
        learning_rate=0.03,    # Slower learning
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    model.train(X_train, y_train)
    print(f"✅ Model trained!")
    
    # Show top custom features
    importance_df = model.get_feature_importance(feature_cols)
    custom_features = [f for f in importance_df['feature'] if 'custom' in f]
    if custom_features:
        print(f"\n   Custom feature importance:")
        for feat in custom_features:
            imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
            print(f"   - {feat}: {imp:.4f}")
    
    # 5. Generate signals with custom confidence
    print("\n[Step 5] Generating signals...")
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    
    # Higher confidence threshold for more selective trading
    signals = model.predict(X_test, confidence_threshold=0.30)
    
    signal_counts = pd.Series(signals).value_counts()
    print(f"✅ Generated {len(signals):,} signals")
    print(f"   Long: {signal_counts.get(1, 0):,}, Neutral: {signal_counts.get(0, 0):,}, Short: {signal_counts.get(-1, 0):,}")
    
    # 6. Backtest with custom parameters
    print("\n[Step 6] Backtesting with custom parameters...")
    
    # Create custom market config
    custom_config = MarketConfig(
        name="custom_topix",
        tick_size=0.5,
        multiplier=10000,
        min_price=1000,
        max_price=5000
    )
    
    backtester = Backtester(
        initial_capital=10_000_000,  # Higher capital
        tick_size=custom_config.tick_size,
        multiplier=custom_config.multiplier
    )
    
    results = backtester.run(
        test_df,
        signals,
        tp_multiplier=4.0,    # Wider TP
        sl_multiplier=0.8,    # Tighter SL
        trail_multiplier=2.0  # Wider trailing stop
    )
    
    # 7. Results
    backtester.print_summary(results)
    
    print(f"\n{'='*70}")
    print("✅ Custom strategy example completed!")
    print(f"{'='*70}")
    
    print("\n💡 CUSTOMIZATION IDEAS:")
    print("   1. Add more custom indicators (order flow, market microstructure)")
    print("   2. Use different label logic (trend-following vs mean-reversion)")
    print("   3. Tune XGBoost hyperparameters (learning_rate, max_depth, etc.)")
    print("   4. Experiment with confidence thresholds (0.15-0.40)")
    print("   5. Adjust TP/SL/Trail multipliers for different risk profiles")
    print("   6. Try different train/test splits or walk-forward validation")
    print("   7. Combine multiple models (ensemble)")
    print("   8. Add market regime filters (trending vs ranging)")
    
    print("\n📚 Next steps:")
    print("   - Run optimizer.py to find optimal parameters")
    print("   - Check Jupyter notebooks for deeper analysis")
    print("   - Read docs/METHODOLOGY.md for strategy details")


if __name__ == '__main__':
    main()
