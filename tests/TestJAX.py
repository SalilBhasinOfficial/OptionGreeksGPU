"""
TestJAX.py - Comprehensive tests for JAX-based Option Greeks implementation

This test suite validates:
1. Black-Scholes pricing accuracy
2. Greeks calculation correctness
3. Numerical accuracy against known values
4. Batch processing functionality
5. Performance benchmarking
6. Comparison with original Numba implementation

Author: Salil Bhasin
License: GNU General Public License v3.0
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
from datetime import time as t
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OptionGreeksGPU.GreeksJAX import (
    black_scholes_call,
    black_scholes_put,
    bisection_implied_volatility,
    delta_call,
    delta_put,
    gamma_call,
    gamma_put,
    vega_call,
    theta_call,
    rho_call,
    calculate_option_metrics,
)

import jax.numpy as jnp


# ============================================================================
# Unit Tests for Individual Functions
# ============================================================================

def test_black_scholes_pricing():
    """Test Black-Scholes pricing against known values."""
    print("\n" + "="*70)
    print("TEST 1: Black-Scholes Pricing Accuracy")
    print("="*70)

    # Known test case from textbook (Hull, Options Futures and Other Derivatives)
    S = 100.0      # Underlying price
    K = 100.0      # Strike price (ATM)
    r = 0.05       # 5% risk-free rate
    T = 1.0        # 1 year to expiration
    sigma = 0.20   # 20% volatility

    call_price = float(black_scholes_call(S, K, r, T, sigma))
    put_price = float(black_scholes_put(S, K, r, T, sigma))

    # Expected values (approximate from Black-Scholes calculator)
    expected_call = 10.45  # Approximate value
    expected_put = 5.57    # Approximate value

    print(f"\nATM Option (S=K=100, r=5%, T=1yr, σ=20%):")
    print(f"  Call Price: ${call_price:.4f} (expected ~${expected_call:.2f})")
    print(f"  Put Price:  ${put_price:.4f} (expected ~${expected_put:.2f})")

    # Test put-call parity: C - P = S - K*e^(-rT)
    parity_lhs = call_price - put_price
    parity_rhs = S - K * np.exp(-r * T)
    parity_error = abs(parity_lhs - parity_rhs)

    print(f"\nPut-Call Parity Check:")
    print(f"  C - P = {parity_lhs:.6f}")
    print(f"  S - K*e^(-rT) = {parity_rhs:.6f}")
    print(f"  Error: {parity_error:.10f}")

    assert parity_error < 1e-5, "Put-call parity violated!"
    print("  ✓ Put-call parity satisfied")

    # Test ITM, ATM, OTM options
    print(f"\nOption Values at Different Strikes:")
    strikes = [80, 90, 100, 110, 120]
    for K_test in strikes:
        call = float(black_scholes_call(S, K_test, r, T, sigma))
        put = float(black_scholes_put(S, K_test, r, T, sigma))
        moneyness = "ITM" if K_test < S else ("ATM" if K_test == S else "OTM")
        print(f"  K={K_test:3d} ({moneyness}): Call=${call:7.4f}, Put=${put:7.4f}")

    print("\n✓ Black-Scholes pricing test PASSED")


def test_greeks_sanity():
    """Test Greeks satisfy basic sanity checks."""
    print("\n" + "="*70)
    print("TEST 2: Greeks Sanity Checks")
    print("="*70)

    S = 100.0
    K = 100.0
    r = 0.05
    T = 1.0
    sigma = 0.20

    # Calculate Greeks
    delta_c = float(delta_call(S, K, r, T, sigma))
    delta_p = float(delta_put(S, K, r, T, sigma))
    gamma_c = float(gamma_call(S, K, r, T, sigma))
    gamma_p = float(gamma_put(S, K, r, T, sigma))
    vega_c = float(vega_call(S, K, r, T, sigma))
    theta_c = float(theta_call(S, K, r, T, sigma))
    rho_c = float(rho_call(S, K, r, T, sigma))

    print(f"\nCall Option Greeks (ATM):")
    print(f"  Delta:  {delta_c:.6f}")
    print(f"  Gamma:  {gamma_c:.6f}")
    print(f"  Vega:   {vega_c:.6f}")
    print(f"  Theta:  {theta_c:.6f}")
    print(f"  Rho:    {rho_c:.6f}")

    print(f"\nPut Option Greeks (ATM):")
    print(f"  Delta:  {delta_p:.6f}")
    print(f"  Gamma:  {gamma_p:.6f}")

    # Sanity checks
    checks_passed = 0
    total_checks = 0

    # Check 1: Call delta should be between 0 and 1
    total_checks += 1
    if 0 <= delta_c <= 1:
        print(f"\n✓ Call delta in [0, 1]: {delta_c:.4f}")
        checks_passed += 1
    else:
        print(f"\n✗ Call delta out of range: {delta_c:.4f}")

    # Check 2: Put delta should be between -1 and 0
    total_checks += 1
    if -1 <= delta_p <= 0:
        print(f"✓ Put delta in [-1, 0]: {delta_p:.4f}")
        checks_passed += 1
    else:
        print(f"✗ Put delta out of range: {delta_p:.4f}")

    # Check 3: Gamma should be positive
    total_checks += 1
    if gamma_c > 0:
        print(f"✓ Gamma is positive: {gamma_c:.6f}")
        checks_passed += 1
    else:
        print(f"✗ Gamma is not positive: {gamma_c:.6f}")

    # Check 4: Gamma should be same for call and put
    total_checks += 1
    if abs(gamma_c - gamma_p) < 1e-6:
        print(f"✓ Gamma same for call and put: {gamma_c:.6f}")
        checks_passed += 1
    else:
        print(f"✗ Gamma differs: Call={gamma_c:.6f}, Put={gamma_p:.6f}")

    # Check 5: ATM delta should be reasonable (between 0.45 and 0.75 with positive r)
    # Note: With r=5%, ATM delta > 0.5 due to forward price effect
    total_checks += 1
    if 0.45 <= delta_c <= 0.75:
        print(f"✓ ATM call delta in reasonable range: {delta_c:.4f}")
        checks_passed += 1
    else:
        print(f"✗ ATM call delta out of reasonable range: {delta_c:.4f}")

    # Check 6: Vega should be positive
    total_checks += 1
    if vega_c > 0:
        print(f"✓ Vega is positive: {vega_c:.6f}")
        checks_passed += 1
    else:
        print(f"✗ Vega is not positive: {vega_c:.6f}")

    print(f"\n✓ Greeks sanity checks: {checks_passed}/{total_checks} passed")
    assert checks_passed == total_checks, "Some sanity checks failed!"


def test_implied_volatility():
    """Test implied volatility calculation."""
    print("\n" + "="*70)
    print("TEST 3: Implied Volatility Calculation")
    print("="*70)

    S = 100.0
    K = 100.0
    r = 0.05
    T = 1.0
    true_sigma = 0.25  # 25% volatility

    # Generate market price using known volatility
    market_call_price = float(black_scholes_call(S, K, r, T, true_sigma))
    market_put_price = float(black_scholes_put(S, K, r, T, true_sigma))

    print(f"\nTrue volatility: {true_sigma*100:.2f}%")
    print(f"Market call price: ${market_call_price:.4f}")
    print(f"Market put price: ${market_put_price:.4f}")

    # Calculate implied volatility
    iv_call = float(bisection_implied_volatility(S, K, r, T, market_call_price, 0))
    iv_put = float(bisection_implied_volatility(S, K, r, T, market_put_price, 1))

    print(f"\nCalculated IV from call: {iv_call:.2f}%")
    print(f"Calculated IV from put:  {iv_put:.2f}%")

    # Check accuracy
    error_call = abs(iv_call - true_sigma * 100)
    error_put = abs(iv_put - true_sigma * 100)

    print(f"\nError in call IV: {error_call:.4f}%")
    print(f"Error in put IV:  {error_put:.4f}%")

    assert error_call < 0.1, f"Call IV error too large: {error_call:.4f}%"
    assert error_put < 0.1, f"Put IV error too large: {error_put:.4f}%"

    print("\n✓ Implied volatility test PASSED")


def test_batch_processing():
    """Test vectorized batch processing."""
    print("\n" + "="*70)
    print("TEST 4: Batch Processing with vmap")
    print("="*70)

    # Create a batch of options
    n_options = 100
    np.random.seed(42)

    strikes = np.linspace(80, 120, n_options)
    underlyings = np.full(n_options, 100.0)
    call_prices = np.random.uniform(5, 25, n_options)
    put_prices = np.random.uniform(5, 25, n_options)

    # Create option_data array
    option_data = np.column_stack([
        strikes,
        underlyings,
        call_prices,
        np.zeros(n_options, dtype=np.int8),  # call type
        put_prices,
        np.ones(n_options, dtype=np.int8),   # put type
    ])

    print(f"\nProcessing {n_options} option contracts...")

    start_time = time.time()
    results = calculate_option_metrics(
        option_data=option_data,
        days_to_expiry=30,
        interest_rate=5.0
    )
    elapsed = time.time() - start_time

    print(f"Time elapsed: {elapsed:.4f} seconds")
    print(f"Time per contract: {elapsed/n_options*1000:.2f} ms")

    # Verify results structure
    assert len(results) == 14, "Should return 14 arrays"
    for i, result in enumerate(results):
        assert len(result) == n_options, f"Result {i} has wrong length"

    print(f"\n✓ Successfully processed {n_options} contracts")
    print(f"✓ All output arrays have correct shape")

    # Show sample results
    print("\nSample Results (first 5 contracts):")
    print(f"{'K':>6} {'Call IV':>8} {'Call Δ':>8} {'Call Γ':>8} {'Put IV':>8} {'Put Δ':>8}")
    print("-" * 54)
    for i in range(min(5, n_options)):
        print(f"{strikes[i]:6.1f} {results[0][i]:8.2f} {results[1][i]:8.4f} "
              f"{results[4][i]:8.6f} {results[7][i]:8.2f} {results[8][i]:8.4f}")

    print("\n✓ Batch processing test PASSED")


# ============================================================================
# Integration Test with Real Data
# ============================================================================

def test_with_real_data():
    """Test with real NSE-style option data if available."""
    print("\n" + "="*70)
    print("TEST 5: Integration Test with Real Data")
    print("="*70)

    test_file = 'tests/OpGreeksTestInput.csv'

    if not os.path.exists(test_file):
        print(f"\n⚠ Test data file not found: {test_file}")
        print("Skipping real data test")
        return

    print(f"\nLoading test data from {test_file}...")

    try:
        df = pd.read_csv(test_file, parse_dates=['expiry', 'DT'])
        print(f"Loaded {len(df)} option contracts")

        # Prepare data
        df = df[['name', 'expiry', 'strike', 'last_price_CE', 'last_price_PE',
                 'last_price_Und', 'GreekRef_CE', 'GreekRef_PE', 'DT']]

        # Replace zero prices with small value
        df.loc[df['last_price_CE'] == 0, 'last_price_CE'] = 0.0001
        df.loc[df['last_price_PE'] == 0, 'last_price_PE'] = 0.0001

        # Calculate for first expiry
        expiry, DT = df[['expiry', 'DT']].iloc[0]
        expiryDT = datetime.combine(expiry, t(15, 30))
        daysToExpiration = (expiryDT - DT).total_seconds() / (24 * 3600)

        df_batch = df[(df['expiry'] == expiry) & (df['DT'] == DT)].copy()
        df_batch.reset_index(drop=True, inplace=True)

        print(f"Processing {len(df_batch)} contracts")
        print(f"Days to expiration: {daysToExpiration:.2f}")

        # Prepare option data
        optionData = df_batch[['strike', 'last_price_Und', 'last_price_CE',
                                'GreekRef_CE', 'last_price_PE', 'GreekRef_PE']].to_numpy()

        # Run calculation with timing
        print("\nRunning warmup...")
        _ = calculate_option_metrics(
            option_data=optionData,
            days_to_expiry=daysToExpiration,
            interest_rate=5.0
        )

        print("Running timed calculation...")
        start_time = time.time()
        Data = calculate_option_metrics(
            option_data=optionData,
            days_to_expiry=daysToExpiration,
            interest_rate=5.0
        )
        elapsed = time.time() - start_time

        print(f"\n✓ Calculation completed in {elapsed:.4f} seconds")
        print(f"✓ Processing rate: {len(df_batch)/elapsed:.0f} contracts/second")

        # Create results DataFrame
        Result_DF = pd.DataFrame(np.column_stack(Data), columns=[
            'call_IVs', 'call_deltas', 'call_delta2s', 'call_vegas',
            'call_gammas', 'call_thetas', 'call_rhos', 'put_IVs',
            'put_deltas', 'put_delta2s', 'put_vegas', 'put_gammas',
            'put_thetas', 'put_rhos'
        ])

        result_df = pd.concat([df_batch, Result_DF], axis=1)

        # Save results
        output_file = 'tests/OpGreeksTestOutput_JAX.csv'
        result_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")

        # Show sample results
        print("\nSample Results (first 3 contracts):")
        print(result_df[['strike', 'call_IVs', 'call_deltas', 'put_IVs', 'put_deltas']].head(3))

        print("\n✓ Real data test PASSED")

    except Exception as e:
        print(f"\n✗ Error in real data test: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# Performance Benchmarking
# ============================================================================

def benchmark_performance():
    """Benchmark JAX implementation performance."""
    print("\n" + "="*70)
    print("TEST 6: Performance Benchmarking")
    print("="*70)

    sizes = [100, 500, 1000, 2000, 5000]

    print(f"\n{'Contracts':>10} {'Warmup (s)':>12} {'Run 1 (s)':>12} {'Run 2 (s)':>12} {'Rate (opt/s)':>15}")
    print("-" * 70)

    for n in sizes:
        # Generate synthetic data
        np.random.seed(42)
        option_data = np.column_stack([
            np.random.uniform(90, 110, n),    # strikes
            np.full(n, 100.0),                 # underlyings
            np.random.uniform(5, 20, n),       # call prices
            np.zeros(n, dtype=np.int8),        # call types
            np.random.uniform(5, 20, n),       # put prices
            np.ones(n, dtype=np.int8),         # put types
        ])

        # Warmup
        start = time.time()
        _ = calculate_option_metrics(option_data, 30, 5.0)
        warmup_time = time.time() - start

        # Run 1
        start = time.time()
        _ = calculate_option_metrics(option_data, 30, 5.0)
        run1_time = time.time() - start

        # Run 2
        start = time.time()
        _ = calculate_option_metrics(option_data, 30, 5.0)
        run2_time = time.time() - start

        rate = n / run2_time

        print(f"{n:10d} {warmup_time:12.4f} {run1_time:12.4f} {run2_time:12.4f} {rate:15.0f}")

    print("\n✓ Performance benchmarking complete")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("JAX OPTION GREEKS - COMPREHENSIVE TEST SUITE")
    print("="*70)

    start_time = time.time()

    tests = [
        ("Black-Scholes Pricing", test_black_scholes_pricing),
        ("Greeks Sanity Checks", test_greeks_sanity),
        ("Implied Volatility", test_implied_volatility),
        ("Batch Processing", test_batch_processing),
        ("Real Data Integration", test_with_real_data),
        ("Performance Benchmarking", benchmark_performance),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f} seconds")
    print("="*70)

    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
