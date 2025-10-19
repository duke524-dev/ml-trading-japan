# Sample Data Guide

This directory contains **sample data** for testing and demonstrations.

## ⚠️ Data Not Included

Due to file size limitations and licensing restrictions, full historical data is **not** included in this repository.

## 📊 Data Requirements

To use this framework, you need OHLCV (Open, High, Low, Close, Volume) data with the following format:

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `datetime` | datetime | Timestamp (YYYY-MM-DD HH:MM:SS) |
| `open` | float | Opening price |
| `high` | float | Highest price in the period |
| `low` | float | Lowest price in the period |
| `close` | float | Closing price |
| `volume` | int | Trading volume |

### Example CSV Format

```csv
datetime,open,high,low,close,volume
2024-01-01 09:00:00,38000,38050,37980,38020,15000
2024-01-01 09:01:00,38020,38070,38010,38060,12000
2024-01-01 09:02:00,38060,38100,38040,38080,18000
...
```

## 🔍 Where to Get Data

### Japanese Stock Indices

**Nikkei225 Mini Futures**:
- Japan Exchange Group (JPX): https://www.jpx.co.jp/
- Market data providers (Bloomberg, Reuters, etc.)
- Broker APIs (Kabu.com, SBI Securities, Rakuten)

**TOPIX Futures**:
- Same sources as Nikkei225
- Data frequency: 1-minute bars recommended

### Data Frequency

- **Minimum**: 1-minute OHLCV bars
- **Recommended**: 1-minute or tick data
- **Period**: At least 6-12 months for meaningful optimization

## 💡 Sample Data Creation

Create your own sample data for testing:

```python
import pandas as pd
import numpy as np

# Generate 1 week of 1-minute data
dates = pd.date_range('2024-01-01 09:00', periods=7*390, freq='1min')
np.random.seed(42)

df = pd.DataFrame({
    'datetime': dates,
    'open': 38000 + np.cumsum(np.random.randn(len(dates)) * 10),
    'high': lambda x: x['open'] + np.random.randint(10, 50, len(dates)),
    'low': lambda x: x['open'] - np.random.randint(10, 50, len(dates)),
    'close': 38000 + np.cumsum(np.random.randn(len(dates)) * 10),
    'volume': np.random.randint(5000, 20000, len(dates))
})

df.to_csv('data/sample_nikkei225.csv', index=False)
```

## 📝 Data Preprocessing

Before using data with this framework:

1. **Check for missing values**: `df.isna().sum()`
2. **Remove duplicates**: `df.drop_duplicates('datetime')`
3. **Sort by time**: `df.sort_values('datetime')`
4. **Validate OHLC**: `high >= open/close >= low`
5. **Check for gaps**: Ensure continuous time series

## ⚖️ Legal & Licensing

- **Respect data licenses**: Check terms of use
- **No redistribution**: Don't share proprietary data
- **Personal use only**: For research and backtesting

---

**Need help with data?** Open an issue on GitHub!
