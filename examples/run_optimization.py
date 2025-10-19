"""
Parameter Optimization Example

Run walk-forward optimization to find best parameters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from ml_framework import WalkForwardOptimizer


def main():
    print("\n" + "="*70)
    print("Parameter Optimization Example")
    print("="*70)
    
    # Load data
    print("\n[Step 1] Loading data...")
    data_path = Path(__file__).parent.parent / "data" / "TOPIX_18months_clean.csv"
    
    if not data_path.exists():
        print(f"⚠️  Data file not found: {data_path}")
        print("   Generating sample data for demonstration...")
        
        dates = pd.date_range('2024-01-01 09:00', periods=10000, freq='1min')
        np.random.seed(42)
        
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
    
    # Create features
    print("\n[Step 2] Creating features...")
    from ml_framework import FeatureEngineer
    
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    df = df.dropna()
    
    print(f"✅ Features ready: {len(df):,} rows")
    
    # Define parameter grid (smaller for demo)
    print("\n[Step 3] Defining parameter grid...")
    param_grid = {
        'profit_threshold': [0.05, 0.08, 0.10],
        'lookahead_bars': [15, 20, 30],
        'ml_confidence': [0.15, 0.20, 0.25],
        'tp_multiplier': [2.0, 3.0, 4.0],
        'sl_multiplier': [1.0, 1.2],
        'trail_multiplier': [1.2, 1.5]
    }
    
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    
    print(f"   Parameter combinations: {total_combinations}")
    for key, values in param_grid.items():
        print(f"   - {key}: {values}")
    
    # Run optimization
    print(f"\n[Step 4] Running walk-forward optimization...")
    print(f"   This may take several minutes...")
    
    optimizer = WalkForwardOptimizer(
        n_splits=3,  # 3-fold for speed (use 5 for production)
        min_trades=10,
        initial_capital=5_000_000,
        tick_size=0.5,
        multiplier=10000
    )
    
    best = optimizer.optimize(df, param_grid, verbose=True)
    
    # Show top 5 results
    print(f"\n{'='*70}")
    print("Top 5 Parameter Sets")
    print(f"{'='*70}")
    
    top_5 = optimizer.get_top_n(5)
    for i, result in enumerate(top_5, 1):
        print(f"\nRank {i}:")
        print(f"  Average ROI: {result['avg_roi']:7.2f}%")
        print(f"  Std Dev:     {result['std_roi']:7.2f}%")
        print(f"  Min ROI:     {result['min_roi']:7.2f}%")
        print(f"  Max ROI:     {result['max_roi']:7.2f}%")
        print(f"  Parameters:")
        for key, value in result['params'].items():
            print(f"    {key:20s}: {value}")
    
    # Save results
    output_path = Path(__file__).parent.parent / "results" / "optimization_results.json"
    optimizer.save_results(str(output_path))
    
    # Plot results
    plot_path = Path(__file__).parent.parent / "visualizations" / "optimization_results.png"
    optimizer.plot_results(top_n=20, save_path=str(plot_path))
    
    print(f"\n{'='*70}")
    print("✅ Optimization completed!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
