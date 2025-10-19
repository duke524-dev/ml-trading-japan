# Frequently Asked Questions (FAQ)

## 📊 Performance Questions

### Q: Is 72.56% ROI real?

**A**: Yes, but with important context:
- ✅ **Real backtested result** on TOPIX data (12 months, 26,182 trades)
- ⚠️ **Backtested only** - NOT live trading results
- ⚠️ **No transaction costs** - Real trading would reduce ROI by 5-15%
- ⚠️ **Perfect execution assumed** - No slippage or partial fills
- ✅ **Walk-Forward validated** - Average 9.16% ROI across 5 folds (more realistic)

**Conclusion**: 72.56% is the best-case scenario. Realistic expectations: 5-15% annual ROI in live trading.

---

### Q: Why did TOPIX perform so much better than Nikkei225?

**A**: Three main reasons:

1. **Trade Frequency**: TOPIX generated 26,182 trades vs Nikkei225's 32 trades (817x more)
2. **Market Characteristics**: TOPIX had lower volatility but more trading opportunities
3. **Parameter Fit**: Optimal parameters for TOPIX (profit_threshold=0.08) differed from Nikkei225 (0.135)

**Key Insight**: Higher trade volume compensated for lower per-trade profit. Different markets require different parameter tuning.

---

### Q: What's the difference between "Quick" and "Optimized" versions?

**A**:

| Aspect | TOPIX Quick | TOPIX Optimized | Improvement |
|--------|-------------|-----------------|-------------|
| **Method** | Borrowed Nikkei225 params | 729 param combinations | N/A |
| **Time** | 15 minutes | 4 hours | 16x longer |
| **ROI** | 6.28% | 72.56% | **11.6x better** |
| **Trades** | 3,238 | 26,182 | 8x more |
| **Win Rate** | 39.3% | 44.13% | +4.8pp |

**Conclusion**: Optimization is crucial. 4 hours of computation = 11.6x performance gain.

---

## 🔧 Technical Questions

### Q: What are the system requirements?

**A**: Minimum requirements:
- **Python**: 3.10 or higher
- **RAM**: 8GB (16GB recommended for optimization)
- **CPU**: Multi-core recommended (optimization uses all cores)
- **Disk**: 2GB for code + data
- **OS**: Windows, macOS, Linux

---

### Q: How long does optimization take?

**A**: Depends on:
- **Quick validation** (existing params): 15-30 minutes
- **Full optimization** (729 combinations × 5 folds): 2-4 hours
- **With multi-core CPU**: Linear speedup (8 cores ≈ 30-60 minutes)

**Tip**: Use `n_jobs=-1` in optimizer to use all CPU cores.

---

### Q: Can I use this for live trading?

**A**: **NOT directly**. Required modifications:

1. ✅ **Add transaction costs**: Spreads, commissions, slippage
2. ✅ **Implement risk management**: Position sizing, account limits
3. ✅ **Paper trade first**: Test in real-time without capital
4. ✅ **Monitor continuously**: Performance can degrade
5. ✅ **Re-optimize periodically**: Markets evolve

**Recommendation**: Use this as a **research framework**, not production trading system.

---

### Q: What data do I need?

**A**: OHLCV (Open, High, Low, Close, Volume) data with:
- **Frequency**: 1-minute bars (minimum)
- **Period**: 6-12 months (minimum for optimization)
- **Format**: CSV with columns: `datetime, open, high, low, close, volume`

**Sources**: JPX, Bloomberg, Reuters, broker APIs. See [data/README.md](../data/README.md).

---

## 🌍 Market Questions

### Q: Can I use this for other markets?

**A**: **Yes!** Framework is market-agnostic. Required steps:

1. Provide OHLCV data in correct format
2. Create market config with tick_size and multiplier
3. Re-optimize parameters (don't use TOPIX/Nikkei225 params)
4. Test thoroughly before live trading

**Example markets**: US indices (S&P 500, NASDAQ), European indices, crypto, forex.

---

### Q: Why Walk-Forward validation instead of regular K-fold?

**A**: Time-series data has **temporal dependencies**:

**Regular K-fold problems**:
- ❌ Train/test not chronological → look-ahead bias
- ❌ Future data influences past training → unrealistic
- ❌ Doesn't simulate real trading progression

**Walk-Forward benefits**:
- ✅ Strictly chronological splits
- ✅ Simulates real trading (train on past, test on future)
- ✅ Tests robustness across different time periods
- ✅ Prevents overfitting

---

## 💰 Investment Questions

### Q: How much capital do I need?

**A**: Depends on market:
- **Nikkei225 Mini**: ¥5,000,000+ recommended (¥3,000,000 minimum)
- **TOPIX**: ¥5,000,000+ recommended
- **High-frequency strategies**: More capital = better risk diversification

**Risk management**: Never risk more than 2% per trade.

---

### Q: What leverage should I use?

**A**: Conservative recommendations:

| Experience Level | Leverage | Expected ROI | Risk Level |
|------------------|----------|--------------|------------|
| Beginner | **1x** | 5-15% | Low |
| Intermediate | **2x** | 10-30% | Medium |
| Advanced | **3x** | 15-45% | High |
| Professional | **3-5x** | 20-60% | Very High |

**Warning**: Higher leverage amplifies BOTH gains AND losses. Start with 1x.

---

## 🤖 ML Model Questions

### Q: Why XGBoost instead of deep learning?

**A**: XGBoost advantages for trading:
- ✅ **Fast training** (minutes vs hours)
- ✅ **Better with tabular data** (technical indicators)
- ✅ **Interpretable** (feature importance)
- ✅ **Less data hungry** (works with 6-12 months)
- ✅ **No GPU required**

Deep learning (LSTM, Transformers) may work but requires more data, compute, and tuning.

---

### Q: How do you prevent overfitting?

**A**: Multiple strategies:

1. ✅ **Walk-Forward validation** - Time-series aware splitting
2. ✅ **Parameter averaging** - Weight across multiple folds
3. ✅ **Out-of-sample testing** - Never train on test data
4. ✅ **XGBoost regularization** - max_depth, min_child_weight
5. ✅ **Feature engineering** - Domain knowledge, not just random features

**Evidence**: All 5 Walk-Forward folds profitable (2.80% - 20.49%).

---

### Q: What features are most important?

**A**: Top features (based on SHAP/feature importance):
1. **Trend**: EMA distances, MACD
2. **Momentum**: RSI, Stochastic
3. **Volatility**: ATR, Bollinger Bands
4. **Volume**: Volume ratios, OBV
5. **Price action**: Returns, ranges

See [METHODOLOGY.md](METHODOLOGY.md) for full list of 63 features.

---

## 🔄 Maintenance Questions

### Q: How often should I re-optimize?

**A**: Recommended schedule:
- **Monthly**: Quick validation check (15 min)
- **Quarterly**: Full re-optimization if performance degrades (4 hours)
- **After market regime change**: Immediately

**Signs to re-optimize**:
- ROI drops > 50% from backtest
- Win rate drops > 10pp
- Sharpe ratio < 1.0

---

### Q: How do I know if the strategy stopped working?

**A**: Warning signs:

| Metric | Backtest | Live Trading | Status |
|--------|----------|--------------|--------|
| ROI | 72.56% | < 10% | ⚠️ Investigate |
| Win Rate | 44.13% | < 35% | ⚠️ Re-optimize |
| Sharpe | ~3.5 | < 1.0 | ⚠️ Stop trading |
| Max DD | 12% | > 25% | 🛑 Stop immediately |

**Action**: Stop trading, analyze cause, re-optimize or modify strategy.

---

## 📝 Code Questions

### Q: How do I customize the strategy?

**A**: Main customization points:

1. **Features**: Add custom indicators in `feature_engineering.py`
2. **ML Model**: Change model in `ml_model.py` (try LightGBM, Random Forest)
3. **Parameters**: Modify parameter space in `optimizer.py`
4. **Risk Management**: Adjust stops in `backtester.py`

See [CONTRIBUTING.md](CONTRIBUTING.md) for code structure.

---

### Q: How do I add a new market?

**A**: Example for S&P 500:

```python
from ml_framework import create_custom_config

# Create S&P 500 E-mini config
sp500_config = create_custom_config(
    name='S&P 500 E-mini',
    tick_size=0.25,
    multiplier=50,
    initial_capital=100000  # $100k USD
)

# Use in backtesting
backtester = Backtester(market_config=sp500_config)
```

---

## 🐛 Troubleshooting

### Q: "ImportError: No module named 'xgboost'"

**A**: Install dependencies:
```bash
pip install -r requirements.txt
```

---

### Q: "ValueError: Data contains NaN/Inf"

**A**: Check data quality:
```python
df = df.dropna()  # Remove NaN
df = df.replace([np.inf, -np.inf], 0)  # Replace infinite
```

---

### Q: Optimization is too slow

**A**: Speed up tips:
1. Use fewer parameter combinations (reduce grid)
2. Use fewer Walk-Forward folds (3 instead of 5)
3. Use shorter data period (6 months instead of 12)
4. Enable multi-core: `n_jobs=-1`

---

### Q: Results don't match paper

**A**: Possible causes:
1. Different data source (check date ranges)
2. Different random seed (set `np.random.seed(42)`)
3. Different library versions (check requirements.txt)
4. Data preprocessing differences

---

## 🤝 Community Questions

### Q: How can I contribute?

**A**: See [CONTRIBUTING.md](CONTRIBUTING.md). Ways to contribute:
- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🧪 Add tests
- 🔬 Test on new markets
- 📊 Share results

---

### Q: Is this project actively maintained?

**A**: Check GitHub for:
- Recent commits
- Open/closed issues
- Pull request activity
- Release notes

---

### Q: Can I use this commercially?

**A**: **Yes!** MIT License allows:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

**Only requirement**: Include original license and copyright notice.

---

## 📞 Still Have Questions?

- **GitHub Issues**: https://github.com/yourusername/ml-trading-japan/issues
- **GitHub Discussions**: https://github.com/yourusername/ml-trading-japan/discussions
- **Email**: [your-email@example.com]

---

**Last Updated**: 2025-10-19
