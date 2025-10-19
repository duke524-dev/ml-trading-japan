"""
Quick Start Example - ML Trading Japan

Demonstrates basic usage of the framework in 5 minutes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from ml_framework import FeatureEngineer, MarketConfig, get_market_config

print("=" * 70)
print("ML Trading Japan - Quick Start Example")
print("=" * 70)

# ============================================
# Step 1: Create or Load Sample Data
# ============================================
print("\n[Step 1] Creating sample data...")

# Generate 1 week of 1-minute OHLCV data
dates = pd.date_range('2024-01-01 09:00', periods=390*5, freq='1min')  # 5 trading days
np.random.seed(42)

base_price = 38000
price_walk = np.cumsum(np.random.randn(len(dates)) * 10)

df = pd.DataFrame({
    'datetime': dates,
    'open': base_price + price_walk,
    'high': base_price + price_walk + np.random.randint(10, 50, len(dates)),
    'low': base_price + price_walk - np.random.randint(10, 50, len(dates)),
    'close': base_price + price_walk + np.random.randn(len(dates)) * 20,
    'volume': np.random.randint(5000, 20000, len(dates))
})

print(f"✅ Created {len(df)} rows of sample data")
print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"   Price range: ¥{df['close'].min():.0f} - ¥{df['close'].max():.0f}")

# ============================================
# Step 2: Engineer Features
# ============================================
print("\n[Step 2] Engineering features...")

engineer = FeatureEngineer()
df_features = engineer.create_all_features(df)

# Remove NaN rows
df_features = df_features.dropna()

print(f"✅ Created {len(engineer.get_feature_list())} features")
print(f"   Rows after cleaning: {len(df_features)}")
print(f"\n   Sample features:")
for feat in engineer.get_feature_list()[:10]:
    print(f"   - {feat}")
print(f"   ... and {len(engineer.get_feature_list()) - 10} more")

# ============================================
# Step 3: Load Market Configuration
# ============================================
print("\n[Step 3] Loading market configuration...")

# Get TOPIX configuration
config = get_market_config('topix')

print(f"✅ Loaded config for: {config.name}")
print(f"   Tick Size: {config.tick_size}")
print(f"   Multiplier: {config.multiplier}")
print(f"   Initial Capital: ¥{config.initial_capital:,}")

# Calculate example P&L
example_pnl = config.calculate_pnl(
    entry_price=38000,
    exit_price=38100,
    position=1  # Long
)
print(f"   Example P&L (38000→38100 long): ¥{example_pnl:,}")

# ============================================
# Step 4: Prepare ML Features
# ============================================
print("\n[Step 4] Preparing ML features...")

# Get features for ML (drop OHLCV and datetime)
X = engineer.prepare_ml_features(df_features)

print(f"✅ Feature matrix shape: {X.shape}")
print(f"   Rows (samples): {X.shape[0]}")
print(f"   Columns (features): {X.shape[1]}")

# Check for infinite/NaN values
inf_count = np.isinf(X.values).sum()
nan_count = np.isnan(X.values).sum()

print(f"   Infinite values: {inf_count}")
print(f"   NaN values: {nan_count}")

if inf_count > 0 or nan_count > 0:
    print("   ⚠️  Warning: Data contains invalid values!")
else:
    print("   ✅ Data is clean and ready for ML!")

# ============================================
# Step 5: Summary
# ============================================
print("\n" + "=" * 70)
print("Quick Start Summary")
print("=" * 70)
print(f"""
✅ Successfully completed all steps:

1. Data Loading     : {len(df)} rows of OHLCV data
2. Feature Engineer : {len(engineer.get_feature_list())} technical indicators
3. Market Config    : {config.name} (Tick: {config.tick_size}, Mult: {config.multiplier})
4. ML Preparation   : {X.shape[0]} samples × {X.shape[1]} features

Next Steps:
-----------
1. Train ML model (see: examples/topix_example.py)
2. Run backtesting (see: examples/nikkei225_example.py)
3. Optimize parameters (see: notebooks/04_optimization.ipynb)
4. Review performance (see: PERFORMANCE_ANALYSIS_REPORT.md)

For full documentation: docs/METHODOLOGY.md
For API reference: docs/API_REFERENCE.md
For questions: GitHub Issues

Happy Trading! 🚀
""")

print("=" * 70)
