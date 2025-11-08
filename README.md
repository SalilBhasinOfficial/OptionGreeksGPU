# OptionGreeksGPU (V3.0.0)

OptionGreeksGPU is a high-performance library for calculating option Greeks using **automatic differentiation** with JAX. The library intelligently selects the best available backend for optimal performance.

## Key Features ✨

- **⚡ Automatic Differentiation**: Greeks computed via JAX's autodiff (no manual formulas!)
- **🚀 Superior Performance**: Up to **42× faster** than traditional implementations
- **🎯 Multi-Backend Support**: Automatically uses JAX → CUDA GPU → CPU (in order of preference)
- **🔧 Unified Codebase**: Single implementation works on CPU, GPU, and TPU
- **✅ Production Ready**: Comprehensive test suite with 100% pass rate

## Performance Benchmarks

Benchmarking results on test data of 824 contracts (included in test directory):

| Backend | Warmup Time | Execution Time | Performance |
|---------|-------------|----------------|-------------|
| CPU Python (baseline) | 144.23s | 221.29s | 1× |
| CPU Numba JIT | 1.94s | 0.68s | 325× faster |
| CUDA GPU Numba | 1.65s | 0.14s | 1,580× faster |
| **JAX (CPU)** | **2.25s** | **0.016s** | **13,830× faster!** |

**JAX Implementation:**
- **49,495 contracts/second** throughput
- Scales linearly to 5,000+ contracts
- 42× faster than Numba CPU, 9× faster than Numba CUDA


Features

	•	Fast Computation: Utilizes GPU acceleration/ Machine code to dramatically reduce computation times for option Greeks.
	•	Easy Integration: Designed to easily integrate with existing Python financial analysis workflows.
	•	Comprehensive: Supports a wide range of Greeks calculations, including Delta, Gamma, Theta, Vega, and Rho.

# Black-Scholes Model

The Black-Scholes model is a fundamental concept in modern financial theory and is widely used for pricing European options on stocks that do not pay dividends. Developed by Fischer Black, Myron Scholes, and Robert Merton in the early 1970s, this model provides a theoretical estimate of the price of European-style options. The beauty of the Black-Scholes model lies in its ability to factor in the major variables affecting option prices, such as the stock price, the exercise price, the risk-free interest rate, the time to expiration, and the volatility of the stock.

The Black-Scholes formula helps investors and traders to determine the fair value of an option, allowing them to make informed trading decisions. Specifically, it calculates the price of an option by inputting the current stock price, the option's strike price, the time until expiration (expressed as a year fraction), the risk-free interest rate, and the volatility of the stock. The model assumes that stock prices follow a lognormal distribution because asset prices cannot be negative, and it considers the constant risk-free rate for the option's life span.

The model is particularly suited for European options, which can only be exercised at expiration, unlike American options, which can be exercised at any time before or at expiration. It's important to note that while the Black-Scholes model provides a robust framework for option valuation, it does have limitations, especially when applied to American options, options on dividend-paying stocks, or in highly volatile markets.


# Installation

## Basic Installation

```bash
pip install OptionGreeksGPU
```

This installs the base package with Numba support (CPU/GPU fallback).

## Recommended: Install with JAX (for best performance)

**For CPU (recommended for most users):**
```bash
pip install OptionGreeksGPU
pip install 'jax[cpu]'
```

**For NVIDIA GPU with CUDA 12.x:**
```bash
pip install OptionGreeksGPU
pip install 'jax[cuda12]'
```

**For NVIDIA GPU with CUDA 11.x:**
```bash
pip install OptionGreeksGPU
pip install 'jax[cuda11]'
```

**For Google Cloud TPU:**
```bash
pip install OptionGreeksGPU
pip install 'jax[tpu]'
```

The library automatically detects and uses the best available backend:
1. **JAX** (if installed) - Best performance, automatic differentiation
2. **CUDA GPU** (if available) - GPU acceleration via Numba
3. **CPU** (fallback) - Numba JIT compilation

# Usage

Here’s a quick example of how to use OptionGreeksGPU (see test.py for exact operation) to compute Greeks for option contracts:

```python
from OptionGreeksGPU.Compute import calculate_option_metrics
import pandas as pd
import numpy as np

### Load your option contracts data (shared in test directory)
df = pd.read_csv('OpGreeksTestInput.csv', parse_dates=['expiry', 'DT'])

### Prepare the data
optionData = df[['strike', 'last_price_Und', 'last_price_CE', 'GreekRef_CE', 'last_price_PE', 'GreekRef_PE']].to_numpy()
interestRate = 5
daysToExpiration = 30  # Example: 30 days to expiration

### Calculate the Greeks
Data = calculate_option_metrics(option_data=optionData, days_to_expiry=daysToExpiration, interest_rate=interestRate)

### Convert the result to a DataFrame
Result_DF = pd.DataFrame(np.column_stack(Data), columns=['call_IVs', 'call_deltas', 'call_delta2s', 'call_vegas', 'call_gammas', 'call_thetas', 'call_rhos', 'put_IVs', 'put_deltas', 'put_delta2s', 'put_vegas', 'put_gammas', 'put_thetas', 'put_rhos'])

### Save or use the results
Result_DF.to_csv('OpGreeksTestOutput.csv')
```

## What's New in Version 3.0.0

### 🎯 JAX Implementation with Automatic Differentiation

Version 3.0 introduces a revolutionary JAX-based backend that uses **automatic differentiation** to compute Greeks:

**Before (Manual Formulas):**
```python
# Had to manually derive and code each Greek formula
def gamma(S, K, r, T, sigma):
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    return norm.pdf(d1) / (S * sigma * sqrt(T))  # Error-prone!
```

**After (Automatic Differentiation):**
```python
# Greeks computed automatically from pricing function
gamma = jax.grad(jax.grad(black_scholes_call, argnums=0), argnums=0)
# Guaranteed mathematically correct!
```

**Benefits:**
- ✅ **No manual derivative formulas** - eliminates human error
- ✅ **Easy to extend** - add new Greeks with one line
- ✅ **Unified codebase** - same code runs on CPU/GPU/TPU
- ✅ **Superior performance** - XLA compilation optimizes everything

### 📊 Performance Improvements

- **49,495 contracts/second** on CPU (vs 1,211 contracts/second previously)
- **42× faster** than Numba CPU implementation
- **9× faster** than Numba CUDA GPU implementation
- Scales linearly to 5,000+ contracts

### 🐛 Bug Fixes

- Fixed undefined variable bug in `GreeksC.py` (CPU fallback)

### 📚 Documentation

- Comprehensive 900+ line technical documentation (`JAX_IMPLEMENTATION.md`)
- Complete test suite with 6 comprehensive tests
- Migration guide and troubleshooting section

### Input Format for OptionGreeksGPU (try using Test.py)
When using the OptionGreeksGPU library to calculate option Greeks based on the Black-Scholes model, the input data should be structured as follows:

input_data = (option_data=[[strikePrices], [underlyingPrices], [callPrices], [callRefs = 0s], [putPrices], [putRefs = 1s]],
                days_to_expiry,
                interest_rate)


strikePrices: An array of strike prices for the options.
underlyingPrices: An array of current prices of the underlying asset.
callPrices: An array of market prices for call options.
callRefs: An array filled with 0s, indicating that the corresponding prices are for call options.
putPrices: An array of market prices for put options.
putRefs: An array filled with 1s, indicating that the corresponding prices are for put options.
interestRate: The risk-free interest rate, expressed as a whole number (e.g., 5 for 5%).
daysToExpiry: The time to expiration of the options, expressed in days (with Decimals).


# Performance

## Benchmark Results (824 contracts)

| Backend | Warmup (s) | Execution (s) | Throughput (ops/s) | Speedup |
|---------|------------|---------------|-------------------|---------|
| CPU Python (baseline) | 144.23 | 221.29 | 3.7 | 1× |
| CPU Numba JIT | 1.94 | 0.68 | 1,211 | 325× |
| CUDA GPU Numba | 1.65 | 0.14 | 5,885 | 1,580× |
| **JAX CPU** | **2.25** | **0.016** | **49,495** | **13,830×** |

## Scaling Performance (JAX)

| Contracts | Time (s) | Throughput (ops/s) |
|-----------|----------|-------------------|
| 100 | 0.003 | 33,455 |
| 500 | 0.013 | 40,004 |
| 1,000 | 0.018 | 54,861 |
| 2,000 | 0.023 | 88,634 |
| **5,000** | **0.039** | **126,838** |

**Key Insights:**
- JAX provides **42× speedup** over Numba CPU
- JAX provides **9× speedup** over Numba CUDA GPU
- Linear scaling: throughput increases with batch size
- Ideal for NSE options (thousands of contracts)

# Contributing

Contributions are welcome! If you’d like to contribute, please fork the repository, create a feature branch, and submit a pull request.

# License

OptionGreeksGPU is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.

# Support

If you encounter any problems or have any suggestions, please open an issue on the project’s GitHub page.
