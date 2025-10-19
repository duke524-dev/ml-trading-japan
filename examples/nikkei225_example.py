"""
Nikkei225 Complete Example

Complete workflow for Nikkei225 Mini futures trading.
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
    get_market_config
)


def main():
    print("\n" + "="*70)
    print("Nikkei225 Complete Trading Example")
    print("="*70)
    
    # 1. Load Nikkei225 data
    print("\n[Step 1] Loading Nikkei225 data...")
    data_path = Path(__file__).parent.parent / "data" / "Nikkei225_18months_clean.csv"
    
    if not data_path.exists():
        print(f"⚠️  Data file not found: {data_path}")
        print("   Please place Nikkei225_18months_clean.csv in data/ folder")
        return
    
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"✅ Loaded {len(df):,} rows from {df['datetime'].min()} to {df['datetime'].max()}")
    
    # 2. Create features
    print("\n[Step 2] Creating 63 technical indicators...")
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    
    initial_len = len(df)
    df = df.dropna()
    print(f"✅ Created features: {initial_len:,} → {len(df):,} rows")
    
    # 3. Prepare features and labels
    print("\n[Step 3] Preparing ML features and labels...")
    
    exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('label')]
    
    # Split train/test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"   Train: {len(train_df):,} rows")
    print(f"   Test:  {len(test_df):,} rows")
    
    # Generate labels (Nikkei225 baseline parameters)
    profit_threshold = 0.05  # 5% profit target
    lookahead_bars = 15
    
    train_df['future_high'] = train_df['high'].rolling(window=lookahead_bars, min_periods=1).max().shift(-lookahead_bars)
    train_df['future_low'] = train_df['low'].rolling(window=lookahead_bars, min_periods=1).min().shift(-lookahead_bars)
    
    train_df['upside'] = (train_df['future_high'] - train_df['close']) / train_df['close']
    train_df['downside'] = (train_df['close'] - train_df['future_low']) / train_df['close']
    
    train_df['label'] = 0
    train_df.loc[train_df['upside'] >= profit_threshold, 'label'] = 1
    train_df.loc[train_df['downside'] >= profit_threshold, 'label'] = -1
    
    train_df = train_df.iloc[:-lookahead_bars]
    
    # 4. Train ML model
    print("\n[Step 4] Training XGBoost model...")
    model = MLSignalPredictor()
    
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train_df['label'].values
    
    model.train(X_train, y_train)
    print(f"✅ Model trained successfully!")
    
    # 5. Generate signals
    print("\n[Step 5] Generating trading signals...")
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    
    ml_confidence = 0.15  # Nikkei225 baseline
    signals = model.predict(X_test, confidence_threshold=ml_confidence)
    
    signal_counts = pd.Series(signals).value_counts()
    print(f"✅ Generated {len(signals):,} signals")
    print(f"   Long: {signal_counts.get(1, 0):,}, Neutral: {signal_counts.get(0, 0):,}, Short: {signal_counts.get(-1, 0):,}")
    
    # 6. Backtest
    print("\n[Step 6] Running backtest...")
    
    config = get_market_config('nikkei225')
    backtester = Backtester(
        initial_capital=5_000_000,
        tick_size=config.tick_size,
        multiplier=config.multiplier
    )
    
    results = backtester.run(
        test_df,
        signals,
        tp_multiplier=2.5,
        sl_multiplier=1.0,
        trail_multiplier=1.5
    )
    
    # 7. Display results
    backtester.print_summary(results)
    
    # Save trades
    trades_df = backtester.get_trades_df()
    if len(trades_df) > 0:
        output_path = Path(__file__).parent.parent / "results" / "nikkei225_example_trades.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(output_path, index=False)
        print(f"\n✅ Trade history saved to: {output_path}")
    
    print(f"\n{'='*70}")
    print("✅ Example completed!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
