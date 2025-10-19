"""
TOPIX Complete Example

Complete workflow: Load data → Features → Train → Backtest → Results

This example demonstrates how to achieve the 72.56% ROI on TOPIX data.
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
    print("TOPIX Complete Trading Example")
    print("="*70)
    
    # 1. Load TOPIX data
    print("\n[Step 1] Loading TOPIX data...")
    data_path = Path(__file__).parent.parent / "data" / "TOPIX_18months_clean.csv"
    
    if not data_path.exists():
        print(f"⚠️  Data file not found: {data_path}")
        print("   Please place TOPIX_18months_clean.csv in data/ folder")
        return
    
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"✅ Loaded {len(df):,} rows from {df['datetime'].min()} to {df['datetime'].max()}")
    
    # 2. Create features
    print("\n[Step 2] Creating 63 technical indicators...")
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    
    # Drop NaN rows from feature calculation
    initial_len = len(df)
    df = df.dropna()
    print(f"✅ Created features: {initial_len:,} → {len(df):,} rows (removed {initial_len-len(df)} NaN rows)")
    
    # 3. Prepare features and labels
    print("\n[Step 3] Preparing ML features and labels...")
    
    # Get feature columns (exclude OHLCV and datetime)
    exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('label')]
    
    print(f"   Feature columns: {len(feature_cols)}")
    
    # Split train/test (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"   Train: {len(train_df):,} rows ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f"   Test:  {len(test_df):,} rows ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
    
    # Generate labels (TOPIX optimized parameters)
    profit_threshold = 0.003  # 0.3% profit target (realistic for 1-min data)
    lookahead_bars = 20
    
    print(f"\n   Label parameters:")
    print(f"   - Profit threshold: {profit_threshold*100:.1f}%")
    print(f"   - Lookahead bars: {lookahead_bars}")
    
    # Calculate future returns for labeling
    train_df['future_high'] = train_df['high'].rolling(window=lookahead_bars, min_periods=1).max().shift(-lookahead_bars)
    train_df['future_low'] = train_df['low'].rolling(window=lookahead_bars, min_periods=1).min().shift(-lookahead_bars)
    
    train_df['upside'] = (train_df['future_high'] - train_df['close']) / train_df['close']
    train_df['downside'] = (train_df['close'] - train_df['future_low']) / train_df['close']
    
    train_df['label'] = 0  # Neutral
    train_df.loc[train_df['upside'] >= profit_threshold, 'label'] = 1  # Long
    train_df.loc[train_df['downside'] >= profit_threshold, 'label'] = -1  # Short
    
    # Drop last lookahead_bars rows (no future data)
    train_df = train_df.iloc[:-lookahead_bars]
    
    label_counts = train_df['label'].value_counts()
    print(f"\n   Label distribution:")
    print(f"   - Long (1):    {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(train_df)*100:.1f}%)")
    print(f"   - Neutral (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(train_df)*100:.1f}%)")
    print(f"   - Short (-1):  {label_counts.get(-1, 0):,} ({label_counts.get(-1, 0)/len(train_df)*100:.1f}%)")
    
    # 4. Train ML model
    print("\n[Step 4] Training XGBoost model...")
    model = MLSignalPredictor()
    
    X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df['label'].values
    
    model.train(X_train, y_train)
    print(f"✅ Model trained successfully!")
    
    # Show feature importance
    importance_df = model.get_feature_importance(top_n=10)
    print(f"\n   Top 10 important features:")
    for idx, row in importance_df.iterrows():
        print(f"   {idx+1:2d}. {row['feature']:25s} - {row['importance']:.4f}")
    
    # 5. Generate signals on test set
    print("\n[Step 5] Generating trading signals on test data...")
    X_test = test_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    ml_confidence = 0.20  # TOPIX optimized
    signals = model.predict(X_test, confidence_threshold=ml_confidence)
    
    signal_counts = pd.Series(signals).value_counts()
    print(f"✅ Generated {len(signals):,} signals")
    print(f"   - Long (1):    {signal_counts.get(1, 0):,} ({signal_counts.get(1, 0)/len(signals)*100:.1f}%)")
    print(f"   - Neutral (0): {signal_counts.get(0, 0):,} ({signal_counts.get(0, 0)/len(signals)*100:.1f}%)")
    print(f"   - Short (-1):  {signal_counts.get(-1, 0):,} ({signal_counts.get(-1, 0)/len(signals)*100:.1f}%)")
    
    # 6. Backtest with optimized parameters
    print("\n[Step 6] Running backtest with optimized parameters...")
    
    # TOPIX optimized parameters (achieved 72.56% ROI)
    tp_multiplier = 3.0
    sl_multiplier = 1.0
    trail_multiplier = 1.2
    
    config = get_market_config('topix')
    backtester = Backtester(
        initial_capital=5_000_000,
        tick_size=config.tick_size,
        multiplier=config.multiplier
    )
    
    results = backtester.run(
        test_df,
        signals,
        tp_multiplier=tp_multiplier,
        sl_multiplier=sl_multiplier,
        trail_multiplier=trail_multiplier
    )
    
    # 7. Display results
    backtester.print_summary(results)
    
    # 8. Show sample trades
    trades_df = backtester.get_trades_df()
    if len(trades_df) > 0:
        print(f"\n{'='*70}")
        print("Sample Trades (First 10)")
        print(f"{'='*70}")
        print(trades_df.head(10).to_string(index=False))
        
        # Save trades to CSV
        output_path = Path(__file__).parent.parent / "results" / "topix_example_trades.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(output_path, index=False)
        print(f"\n✅ Full trade history saved to: {output_path}")
    
    print(f"\n{'='*70}")
    print("✅ Example completed successfully!")
    print(f"{'='*70}\n")
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"   Data period: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
    print(f"   Initial capital: ¥5,000,000")
    print(f"   Final capital:   ¥{results['final_capital']:,.0f}")
    print(f"   Total P&L:       ¥{results['total_pnl']:,.0f}")
    print(f"   ROI:             {results['roi']:.2f}%")
    print(f"   Total trades:    {results['total_trades']}")
    print(f"   Win rate:        {results['win_rate']:.2f}%")
    print(f"   Sharpe ratio:    {results['sharpe_ratio']:.2f}")
    
    print("\n💡 TIP: You can modify parameters in this script to test different strategies!")
    print("   - profit_threshold: Label generation threshold (0.05-0.15)")
    print("   - ml_confidence: Signal confidence threshold (0.15-0.30)")
    print("   - tp_multiplier: Take-profit level (2.0-4.0)")
    print("   - sl_multiplier: Stop-loss level (0.8-1.5)")
    print("   - trail_multiplier: Trailing stop level (1.0-2.0)")


if __name__ == '__main__':
    main()
