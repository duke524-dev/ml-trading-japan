# Contributing to ML Trading Japan

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Types of Contributions

We welcome:
- 🐛 **Bug reports** - Help us find and fix issues
- 💡 **Feature requests** - Suggest new capabilities
- 📝 **Documentation improvements** - Make docs clearer
- 🧪 **Test cases** - Improve code coverage
- 🔬 **New market tests** - Validate on different indices
- 📊 **Performance analysis** - Share your results
- 🎨 **Code improvements** - Refactor for better quality

---

## 🐛 Reporting Bugs

### Before Reporting

1. Check [existing issues](https://github.com/yourusername/ml-trading-japan/issues)
2. Try latest version
3. Read [FAQ](docs/FAQ.md)

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With data '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Windows 11]
- Python version: [e.g., 3.10.5]
- Library versions: Run `pip freeze`

**Additional context**
Add any other context, screenshots, or error messages.
```

---

## 💡 Suggesting Features

### Feature Request Template

```markdown
**Problem Statement**
Describe the problem this feature would solve.

**Proposed Solution**
Describe your proposed solution.

**Alternatives Considered**
What alternatives have you considered?

**Additional Context**
Add any other context, mockups, or examples.
```

---

## 📝 Code Contributions

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/ml-trading-japan.git
   cd ml-trading-japan
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in editable mode
   ```

4. **Install dev dependencies**
   ```bash
   pip install pytest black flake8 mypy
   ```

### Code Style

We follow **PEP 8** with these specifics:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Single quotes for strings (except docstrings)
- **Imports**: Organized (stdlib, third-party, local)

### Formatting Tools

```bash
# Auto-format code
black ml_framework/ examples/ tests/

# Check style
flake8 ml_framework/ examples/ tests/ --max-line-length=100

# Type checking
mypy ml_framework/
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: Add support for US stock indices
fix: Correct ATR calculation in volatile markets
docs: Update installation instructions
test: Add unit tests for FeatureEngineer
refactor: Simplify backtester logic
perf: Optimize Walk-Forward loop
```

### Pull Request Process

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes**
   - Write code
   - Add tests
   - Update documentation
   - Run tests: `pytest`

3. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: Add your feature"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Go to GitHub
   - Click "New Pull Request"
   - Fill in template (see below)
   - Wait for review

### Pull Request Template

```markdown
**Description**
Brief description of changes.

**Motivation and Context**
Why is this change needed? What problem does it solve?

**How Has This Been Tested?**
Describe testing methodology.

**Types of changes**
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

**Checklist:**
- [ ] Code follows project style guidelines
- [ ] Added tests for new functionality
- [ ] All tests pass locally
- [ ] Updated documentation
- [ ] Added type hints
- [ ] No breaking changes (or documented if necessary)
```

---

## 🧪 Testing Guidelines

### Writing Tests

Use **pytest** for all tests:

```python
# tests/test_feature_engineering.py
import pytest
import pandas as pd
from ml_framework import FeatureEngineer

def test_feature_engineer_basic():
    """Test basic feature creation."""
    df = create_sample_data()
    engineer = FeatureEngineer()
    
    result = engineer.create_all_features(df)
    
    assert 'rsi_14' in result.columns
    assert 'ema_20' in result.columns
    assert len(engineer.get_feature_list()) > 60

def test_feature_engineer_handles_nan():
    """Test NaN handling."""
    df = create_sample_data_with_nan()
    engineer = FeatureEngineer()
    
    result = engineer.create_all_features(df)
    result = result.dropna()
    
    assert not result.isnull().any().any()
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_feature_engineering.py

# Run with coverage
pytest --cov=ml_framework tests/

# Run with verbose output
pytest -v
```

### Test Coverage

Aim for **80%+ coverage** on new code:

```bash
pytest --cov=ml_framework --cov-report=html
# Open htmlcov/index.html
```

---

## 📚 Documentation Guidelines

### Docstring Format

Use **Google-style docstrings**:

```python
def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    RSI measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.
    
    Args:
        series: Price series (typically close prices)
        period: Number of periods for RSI calculation (typically 14)
        
    Returns:
        Series with RSI values (0-100)
        
    Raises:
        ValueError: If period < 1
        
    Example:
        >>> rsi = calculate_rsi(df['close'], period=14)
        >>> print(rsi.tail())
        
    References:
        https://www.investopedia.com/terms/r/rsi.asp
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    
    # Implementation...
```

### Type Hints

**Always** add type hints:

```python
from typing import List, Dict, Optional, Tuple

def process_trades(
    trades: List[Dict[str, float]],
    capital: float,
    max_positions: Optional[int] = None
) -> Tuple[float, List[str]]:
    """Process trades and return final capital and warnings."""
    # Implementation...
```

### Documentation Files

- **README.md**: High-level overview
- **METHODOLOGY.md**: Technical details
- **API_REFERENCE.md**: Code documentation
- **FAQ.md**: Common questions
- **Examples**: Runnable scripts

---

## 🔬 Testing on New Markets

Want to test on different markets? Great!

### Required Information

1. **Market name**: e.g., "S&P 500 E-mini"
2. **Tick size**: Minimum price movement
3. **Multiplier**: Contract multiplier
4. **Data period**: At least 6 months
5. **Results**: ROI, win rate, trades, etc.

### Sharing Results

Create an issue with:
- Market details
- Data period and frequency
- Parameter settings
- Performance metrics
- Any observations

---

## 🎨 Code Review Process

### What We Look For

- ✅ **Correctness**: Does it work?
- ✅ **Tests**: Is it tested?
- ✅ **Documentation**: Is it documented?
- ✅ **Style**: Follows PEP 8?
- ✅ **Performance**: Is it efficient?
- ✅ **Clarity**: Is it readable?

### Review Timeline

- **Initial review**: Within 3-5 days
- **Follow-up**: Within 2 days
- **Merge**: After approval + CI passing

### Addressing Feedback

- Be receptive to suggestions
- Ask questions if unclear
- Make requested changes
- Mark conversations as resolved

---

## 📊 Benchmarking

### Performance Benchmarks

When optimizing code, provide benchmarks:

```python
import time

# Before
start = time.time()
result_old = old_implementation(data)
time_old = time.time() - start

# After
start = time.time()
result_new = new_implementation(data)
time_new = time.time() - start

print(f"Old: {time_old:.2f}s, New: {time_new:.2f}s")
print(f"Speedup: {time_old/time_new:.1f}x")
```

---

## 🏆 Recognition

Contributors will be:
- ✅ Listed in [CONTRIBUTORS.md](CONTRIBUTORS.md)
- ✅ Mentioned in release notes
- ✅ Credited in related publications (if applicable)

---

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all.

### Our Standards

**Positive behavior**:
- ✅ Being respectful of differing viewpoints
- ✅ Gracefully accepting constructive criticism
- ✅ Focusing on what is best for the community
- ✅ Showing empathy towards other community members

**Unacceptable behavior**:
- ❌ Trolling, insulting/derogatory comments
- ❌ Public or private harassment
- ❌ Publishing others' private information
- ❌ Other conduct which could reasonably be considered inappropriate

### Enforcement

Violations can be reported to [your-email@example.com]. All complaints will be reviewed and investigated.

---

## ❓ Questions?

- **General questions**: GitHub Discussions
- **Technical questions**: GitHub Issues
- **Private matters**: Email [your-email@example.com]

---

Thank you for contributing to ML Trading Japan! 🚀

**Last Updated**: 2025-10-19
