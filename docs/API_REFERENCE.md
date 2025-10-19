# API Reference

## ml_framework

### FeatureEngineer

创建63个技术指标的核心类。

```python
from ml_framework import FeatureEngineer

engineer = FeatureEngineer()
df = engineer.create_all_features(df)
```

**方法**:
- `create_all_features(df: pd.DataFrame) -> pd.DataFrame`: 创建所有63个技术指标

---

### MLSignalPredictor

XGBoost 3分类交易信号预测器。

```python
from ml_framework import MLSignalPredictor

model = MLSignalPredictor(n_estimators=200, max_depth=6, learning_rate=0.05)
model.train(X_train, y_train)
signals = model.predict(X_test, confidence_threshold=0.20)
```

**参数**:
- `n_estimators`: 树的数量 (default: 200)
- `max_depth`: 树的最大深度 (default: 6)
- `learning_rate`: 学习率 (default: 0.05)

**方法**:
- `train(X, y, verbose=True)`: 训练模型
- `predict(X, confidence_threshold=0.20)`: 生成信号 (1=Long, 0=Neutral, -1=Short)
- `get_feature_importance(top_n=20)`: 获取特征重要性

---

### Backtester

向量化回测引擎，支持ATR动态止损。

```python
from ml_framework import Backtester

backtester = Backtester(initial_capital=5_000_000, tick_size=0.5, multiplier=10000)
results = backtester.run(df, signals, tp_multiplier=3.0, sl_multiplier=1.0, trail_multiplier=1.2)
```

**参数**:
- `initial_capital`: 初始资金
- `tick_size`: 最小变动单位
- `multiplier`: 合约乘数

**方法**:
- `run(df, signals, tp_multiplier, sl_multiplier, trail_multiplier)`: 运行回测
- `get_trades_df()`: 获取交易记录DataFrame
- `print_summary(results)`: 打印结果摘要

**返回指标**:
- `roi`: 投资回报率 (%)
- `total_trades`: 交易总数
- `win_rate`: 胜率 (%)
- `profit_factor`: 盈利因子
- `sharpe_ratio`: 夏普比率
- `max_drawdown_pct`: 最大回撤 (%)

---

### WalkForwardOptimizer

Walk-Forward 参数优化器。

```python
from ml_framework import WalkForwardOptimizer

param_grid = {
    'profit_threshold': [0.05, 0.08, 0.10],
    'lookahead_bars': [15, 20, 30],
    'ml_confidence': [0.15, 0.20, 0.25],
    'tp_multiplier': [2.0, 3.0, 4.0],
    'sl_multiplier': [0.8, 1.0, 1.2],
    'trail_multiplier': [1.0, 1.2, 1.5]
}

optimizer = WalkForwardOptimizer(n_splits=5, min_trades=10)
best = optimizer.optimize(df, param_grid)
```

**方法**:
- `optimize(df, param_grid, verbose=True)`: 运行优化
- `get_top_n(n=10)`: 获取Top N参数集
- `save_results(filepath)`: 保存结果到JSON
- `plot_results(top_n=20)`: 可视化结果

---

### MarketConfig

市场配置管理。

```python
from ml_framework import get_market_config

config = get_market_config('topix')  # 或 'nikkei225'
```

**属性**:
- `name`: 市场名称
- `tick_size`: 最小变动单位
- `multiplier`: 合约乘数
- `initial_capital`: 初始资金

**方法**:
- `calculate_pnl(entry_price, exit_price, position)`: 计算盈亏

---

## 完整示例

查看 `examples/` 目录获取完整使用示例。
