# Methodology - 技术方法论

## 系统架构

ML Trading Japan 使用**机器学习 + 规则系统**混合架构：

```
原始数据 → 特征工程 → ML模型 → 交易信号 → 风险管理 → 回测验证
  (OHLCV)    (63指标)   (XGBoost)  (Long/Short)  (TP/SL/Trail)  (Performance)
```

---

## 1. 特征工程 (Feature Engineering)

### 63个技术指标分类

#### 趋势指标 (Trend)
- **EMA** (5, 10, 20, 50): 指数移动平均
- **MACD**: 移动平均收敛发散
- **ADX**: 平均趋向指标

#### 动量指标 (Momentum)
- **RSI** (7, 14, 21): 相对强弱指标
- **Stochastic**: 随机指标
- **CCI**: 商品通道指标
- **Williams %R**: 威廉指标
- **ROC**: 变化率

#### 波动率指标 (Volatility)
- **ATR** (7, 14, 21): 真实波幅
- **Bollinger Bands** (10, 20): 布林带

#### 成交量指标 (Volume)
- **OBV**: 能量潮
- **Volume SMA**: 成交量移动平均

#### 市场微观结构 (Microstructure)
- 价格比率 (high/low, close/open)
- 影线分析 (upper shadow, lower shadow)
- K线形态 (body ratio)

---

## 2. 标签生成 (Label Generation)

使用**前瞻性价格变动**生成三分类标签：

```python
# 计算未来N根K线的最高/最低价
future_high = df['high'].rolling(lookahead_bars).max().shift(-lookahead_bars)
future_low = df['low'].rolling(lookahead_bars).min().shift(-lookahead_bars)

# 计算潜在收益
upside = (future_high - close) / close
downside = (close - future_low) / close

# 生成标签
if upside >= profit_threshold:
    label = 1  # Long
elif downside >= profit_threshold:
    label = -1  # Short
else:
    label = 0  # Neutral
```

**参数说明**:
- `profit_threshold`: 利润阈值 (TOPIX优化值: 0.08 = 8%)
- `lookahead_bars`: 向前看的K线数 (优化值: 20根)

---

## 3. 机器学习模型 (ML Model)

### XGBoost 配置

```python
XGBClassifier(
    n_estimators=200,      # 树的数量
    max_depth=6,           # 树深度
    learning_rate=0.05,    # 学习率
    subsample=0.8,         # 样本抽样比例
    colsample_bytree=0.8,  # 特征抽样比例
    random_state=42
)
```

### 为什么选择XGBoost？

1. **非线性关系**: 能捕捉技术指标间的复杂交互
2. **特征重要性**: 自动识别最有价值的指标
3. **过拟合控制**: 通过参数调节防止过拟合
4. **速度快**: 适合大规模数据

### 三分类策略

- **Class 0 (Short)**: 预测价格下跌
- **Class 1 (Neutral)**: 不确定，不交易
- **Class 2 (Long)**: 预测价格上涨

使用**置信度阈值**过滤低置信度信号：
```python
signals = model.predict(X, confidence_threshold=0.20)
```

---

## 4. 风险管理 (Risk Management)

### ATR动态止损

所有止损/止盈基于**ATR (真实波幅)**动态调整：

```python
atr = current_bar['atr_14']

# 多头
take_profit = entry_price + (tp_multiplier * atr)
stop_loss = entry_price - (sl_multiplier * atr)
trailing_stop = highest_price - (trail_multiplier * atr)

# 空头
take_profit = entry_price - (tp_multiplier * atr)
stop_loss = entry_price + (sl_multiplier * atr)
trailing_stop = lowest_price + (trail_multiplier * atr)
```

**TOPIX优化参数**:
- `tp_multiplier = 3.0`: 止盈 = 3倍ATR
- `sl_multiplier = 1.0`: 止损 = 1倍ATR
- `trail_multiplier = 1.2`: 追踪止损 = 1.2倍ATR

### 退出条件优先级

1. **TP (Take Profit)**: 达到止盈目标
2. **SL (Stop Loss)**: 触发止损
3. **Trail**: 触发追踪止损
4. **Signal**: 反向信号出现
5. **EOD**: 数据结束

---

## 5. Walk-Forward 验证

使用**时间序列交叉验证**避免未来信息泄露：

```
Fold 1: |--Train--|--Test--|
Fold 2:     |--Train--|--Test--|
Fold 3:         |--Train--|--Test--|
Fold 4:             |--Train--|--Test--|
Fold 5:                 |--Train--|--Test--|
```

### TOPIX 5-Fold 结果

| Fold | ROI | 交易数 | 胜率 |
|------|-----|--------|------|
| 1 | 8.34% | 4,231 | 42.1% |
| 2 | 9.12% | 4,587 | 43.8% |
| 3 | 11.28% | 5,012 | 45.2% |
| 4 | 8.76% | 4,398 | 41.9% |
| 5 | 8.32% | 4,954 | 42.5% |
| **平均** | **9.16%** | **4,636** | **43.1%** |

**所有折都盈利** ✅ → 策略稳健

---

## 6. 性能指标

### 关键指标定义

- **ROI (Return on Investment)**: 投资回报率 = (最终资金 - 初始资金) / 初始资金 × 100%
- **胜率 (Win Rate)**: 盈利交易数 / 总交易数 × 100%
- **盈利因子 (Profit Factor)**: 总盈利 / 总亏损
- **夏普比率 (Sharpe Ratio)**: (平均收益 - 无风险利率) / 收益标准差 × √252
- **最大回撤 (Max Drawdown)**: 从峰值到谷底的最大跌幅

### TOPIX优化结果

| 指标 | 值 |
|------|------|
| ROI | **72.56%** |
| 总交易数 | 26,182 |
| 胜率 | 44.13% |
| 盈利因子 | 1.68 |
| 夏普比率 | 1.95 |
| 最大回撤 | -8.23% |

---

## 7. 关键假设

1. **滑点**: 未考虑（保守估计）
2. **手续费**: 未考虑（实际会降低收益）
3. **流动性**: 假设订单立即成交
4. **执行延迟**: 假设信号即时执行
5. **市场冲击**: 假设订单不影响价格

**实盘部署需考虑这些因素！**

---

## 8. 过拟合风险控制

### 防止过拟合的措施

1. ✅ **Walk-Forward验证**: 时间序列交叉验证
2. ✅ **参数网格搜索**: 系统化搜索而非手工调参
3. ✅ **多折验证**: 确保参数在不同时期稳健
4. ✅ **特征选择**: XGBoost自动识别重要特征
5. ✅ **正则化**: XGBoost内置L1/L2正则化

### 识别过拟合的信号

⚠️ **警告信号**:
- 训练集准确率 >> 测试集准确率
- 不同Fold之间ROI差异 > 50%
- 最佳参数过于极端（边界值）
- 回测ROI >> Walk-Forward ROI

---

## 9. 进一步优化方向

### 短期优化 (1-2周)
- 📊 集成学习 (Ensemble): XGBoost + LightGBM + CatBoost
- 🎯 动态仓位管理: 基于置信度调整仓位
- 🔄 多时间框架: 结合5分钟、15分钟信号

### 中期优化 (1-3个月)
- 🧠 深度学习: LSTM/Transformer捕捉时序模式
- 📈 市场状态分类: 趋势/震荡/反转状态识别
- 💹 情绪指标: 整合市场情绪数据

### 长期研究 (3-6个月)
- 🤖 强化学习: Deep Q-Network优化交易决策
- 🌐 多市场联动: 跨市场套利策略
- 📡 高频数据: Tick级别微观结构分析

---

## 参考文献

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
2. Prado, M. L. (2018). Advances in Financial Machine Learning.
3. Jansen, S. (2020). Machine Learning for Algorithmic Trading.

---

**更新日期**: 2025-10-19
