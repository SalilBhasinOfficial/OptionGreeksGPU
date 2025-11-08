"""
GreeksJAX.py - High-performance Option Greeks computation using JAX

This module implements Black-Scholes option pricing and Greeks calculation
using JAX library for automatic differentiation and GPU/CPU acceleration.

Key Features:
- Automatic differentiation for Greeks (no manual formulas needed)
- Unified CPU/GPU/TPU execution
- Vectorized batch processing with jax.vmap
- XLA compilation for optimal performance
- Clean, maintainable code

Author: Salil Bhasin
License: GNU General Public License v3.0
"""

import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.stats import norm
from jax.scipy.optimize import minimize
import numpy as np


# ============================================================================
# Black-Scholes Pricing Functions
# ============================================================================

@jit
def norm_cdf(x):
    """
    Standard normal cumulative distribution function.

    Args:
        x: Input value

    Returns:
        Probability P(X <= x) where X ~ N(0,1)
    """
    return norm.cdf(x)


@jit
def norm_pdf(x):
    """
    Standard normal probability density function.

    Args:
        x: Input value

    Returns:
        Density at x for N(0,1)
    """
    return norm.pdf(x)


@jit
def black_scholes_d1_d2(S, K, r, T, sigma):
    """
    Calculate d1 and d2 parameters for Black-Scholes formula.

    Args:
        S: Underlying asset price (spot price)
        K: Strike price
        r: Risk-free interest rate (annualized)
        T: Time to expiration (in years)
        sigma: Volatility (annualized, as decimal not percentage)

    Returns:
        Tuple of (d1, d2)
    """
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return d1, d2


@jit
def black_scholes_call(S, K, r, T, sigma):
    """
    Black-Scholes call option pricing formula.

    Args:
        S: Underlying asset price
        K: Strike price
        r: Risk-free interest rate (annualized)
        T: Time to expiration (in years)
        sigma: Volatility (annualized, as decimal)

    Returns:
        Call option theoretical price
    """
    # Handle edge case: at expiration
    def at_expiry():
        return jnp.maximum(S - K, 0.0)

    def before_expiry():
        d1, d2 = black_scholes_d1_d2(S, K, r, T, sigma)
        return S * norm_cdf(d1) - K * jnp.exp(-r * T) * norm_cdf(d2)

    return jnp.where(T <= 0, at_expiry(), before_expiry())


@jit
def black_scholes_put(S, K, r, T, sigma):
    """
    Black-Scholes put option pricing formula.

    Args:
        S: Underlying asset price
        K: Strike price
        r: Risk-free interest rate (annualized)
        T: Time to expiration (in years)
        sigma: Volatility (annualized, as decimal)

    Returns:
        Put option theoretical price
    """
    # Handle edge case: at expiration
    def at_expiry():
        return jnp.maximum(K - S, 0.0)

    def before_expiry():
        d1, d2 = black_scholes_d1_d2(S, K, r, T, sigma)
        return K * jnp.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

    return jnp.where(T <= 0, at_expiry(), before_expiry())


# ============================================================================
# Implied Volatility Calculation
# ============================================================================

@jit
def implied_volatility_objective(sigma, S, K, r, T, market_price, option_type):
    """
    Objective function for implied volatility calculation.
    Computes squared error between model price and market price.

    Args:
        sigma: Volatility (to be optimized)
        S: Underlying price
        K: Strike price
        r: Risk-free rate
        T: Time to expiration
        market_price: Observed market price
        option_type: 0 for call, 1 for put

    Returns:
        Squared error between theoretical and market price
    """
    theoretical_price = jnp.where(
        option_type == 0,
        black_scholes_call(S, K, r, T, sigma),
        black_scholes_put(S, K, r, T, sigma)
    )
    return (theoretical_price - market_price) ** 2


@jit
def bisection_implied_volatility(S, K, r, T, market_price, option_type,
                                   tol=1e-6, max_iter=100):
    """
    Calculate implied volatility using bisection method.

    This method is robust and guaranteed to converge if a solution exists
    within the search bounds.

    Args:
        S: Underlying price
        K: Strike price
        r: Risk-free rate
        T: Time to expiration
        market_price: Market price of option
        option_type: 0 for call, 1 for put
        tol: Tolerance for convergence
        max_iter: Maximum iterations

    Returns:
        Implied volatility (as percentage)
    """
    # Handle at-expiry case
    def at_expiry():
        return 0.01  # Return minimal volatility

    def before_expiry():
        # Define pricing function for this specific option
        def price_at_vol(sigma):
            return jnp.where(
                option_type == 0,
                black_scholes_call(S, K, r, T, sigma / 100.0),
                black_scholes_put(S, K, r, T, sigma / 100.0)
            )

        # Bisection search between 0.001% and 500%
        def bisection_step(carry):
            low, high, i = carry
            mid = (low + high) / 2.0

            # Ensure minimum volatility
            mid = jnp.maximum(mid, 0.001)

            price_mid = price_at_vol(mid)

            # Update bounds
            new_low = jnp.where(price_mid < market_price, mid, low)
            new_high = jnp.where(price_mid >= market_price, mid, high)

            return (new_low, new_high, i + 1)

        def continue_condition(carry):
            low, high, i = carry
            return (high - low > tol) & (i < max_iter)

        # Run bisection
        initial_carry = (0.001, 500.0, 0)
        final_low, final_high, _ = jax.lax.while_loop(
            continue_condition,
            bisection_step,
            initial_carry
        )

        return (final_low + final_high) / 2.0

    return jnp.where(T <= 0, at_expiry(), before_expiry())


# Vectorize IV calculation for batch processing
batch_implied_volatility = vmap(bisection_implied_volatility, in_axes=(0, 0, None, None, 0, 0))


# ============================================================================
# Greeks Computation via Automatic Differentiation
# ============================================================================

# First-order Greeks
delta_call = jit(grad(black_scholes_call, argnums=0))    # ∂C/∂S
delta_put = jit(grad(black_scholes_put, argnums=0))      # ∂P/∂S

vega_call = jit(grad(black_scholes_call, argnums=4))     # ∂C/∂σ
vega_put = jit(grad(black_scholes_put, argnums=4))       # ∂P/∂σ

theta_call = jit(grad(black_scholes_call, argnums=3))    # ∂C/∂T
theta_put = jit(grad(black_scholes_put, argnums=3))      # ∂P/∂T

rho_call = jit(grad(black_scholes_call, argnums=2))      # ∂C/∂r
rho_put = jit(grad(black_scholes_put, argnums=2))        # ∂P/∂r

# Second-order Greeks
gamma_call = jit(grad(grad(black_scholes_call, argnums=0), argnums=0))  # ∂²C/∂S²
gamma_put = jit(grad(grad(black_scholes_put, argnums=0), argnums=0))    # ∂²P/∂S²

# Delta2 (strike sensitivity) - custom Greek
delta2_call = jit(grad(black_scholes_call, argnums=1))   # ∂C/∂K
delta2_put = jit(grad(black_scholes_put, argnums=1))     # ∂P/∂K


@jit
def compute_all_greeks_call(S, K, r, T, sigma):
    """
    Compute all Greeks for a call option.

    Args:
        S: Underlying price
        K: Strike price
        r: Risk-free rate
        T: Time to expiration
        sigma: Volatility (as decimal)

    Returns:
        Tuple of (delta, delta2, vega, gamma, theta, rho)
    """
    delta = delta_call(S, K, r, T, sigma)
    delta2 = delta2_call(S, K, r, T, sigma)
    vega = vega_call(S, K, r, T, sigma) / 100.0  # Scale to 1% vol change
    gamma = gamma_call(S, K, r, T, sigma)
    theta = theta_call(S, K, r, T, sigma) / 365.0  # Scale to per-day
    rho = rho_call(S, K, r, T, sigma) / 100.0  # Scale to 1% rate change

    return delta, delta2, vega, gamma, theta, rho


@jit
def compute_all_greeks_put(S, K, r, T, sigma):
    """
    Compute all Greeks for a put option.

    Args:
        S: Underlying price
        K: Strike price
        r: Risk-free rate
        T: Time to expiration
        sigma: Volatility (as decimal)

    Returns:
        Tuple of (delta, delta2, vega, gamma, theta, rho)
    """
    delta = delta_put(S, K, r, T, sigma)
    delta2 = delta2_put(S, K, r, T, sigma)
    vega = vega_put(S, K, r, T, sigma) / 100.0  # Scale to 1% vol change
    gamma = gamma_put(S, K, r, T, sigma)
    theta = theta_put(S, K, r, T, sigma) / 365.0  # Scale to per-day
    rho = rho_put(S, K, r, T, sigma) / 100.0  # Scale to 1% rate change

    return delta, delta2, vega, gamma, theta, rho


# Vectorize Greeks computation for batch processing
batch_greeks_call = vmap(compute_all_greeks_call, in_axes=(0, 0, None, None, 0))
batch_greeks_put = vmap(compute_all_greeks_put, in_axes=(0, 0, None, None, 0))


# ============================================================================
# Main Calculation Function (API-compatible with original implementation)
# ============================================================================

def calculate_option_metrics(option_data, days_to_expiry, interest_rate):
    """
    Calculate option Greeks for a batch of option contracts.

    This function maintains API compatibility with the original Numba implementation
    while using JAX for improved performance and automatic differentiation.

    Args:
        option_data: NumPy array of shape (N, 6) containing:
            - Column 0: Strike prices
            - Column 1: Underlying prices
            - Column 2: Call market prices
            - Column 3: Call option type flags (0 for call)
            - Column 4: Put market prices
            - Column 5: Put option type flags (1 for put)
        days_to_expiry: Days until option expiration (scalar)
        interest_rate: Risk-free interest rate as percentage (e.g., 5 for 5%)

    Returns:
        List of 14 NumPy arrays containing:
        [call_IVs, call_deltas, call_delta2s, call_vegas, call_gammas,
         call_thetas, call_rhos, put_IVs, put_deltas, put_delta2s,
         put_vegas, put_gammas, put_thetas, put_rhos]
    """
    # Convert inputs to proper format
    r = interest_rate / 100.0  # Convert percentage to decimal
    T = days_to_expiry / 365.0  # Convert days to years

    # Extract data columns
    K = jnp.array(option_data[:, 0], dtype=jnp.float32)  # Strike prices
    S = jnp.array(option_data[:, 1], dtype=jnp.float32)  # Underlying prices

    call_prices = jnp.array(option_data[:, 2], dtype=jnp.float32)
    call_types = jnp.array(option_data[:, 3], dtype=jnp.int8)

    put_prices = jnp.array(option_data[:, 4], dtype=jnp.float32)
    put_types = jnp.array(option_data[:, 5], dtype=jnp.int8)

    # Calculate implied volatilities for all options
    call_IVs = batch_implied_volatility(S, K, r, T, call_prices, call_types)
    put_IVs = batch_implied_volatility(S, K, r, T, put_prices, put_types)

    # Convert IV percentages to decimals for Greeks calculation
    call_sigmas = call_IVs / 100.0
    put_sigmas = put_IVs / 100.0

    # Calculate all Greeks using automatic differentiation
    call_delta, call_delta2, call_vega, call_gamma, call_theta, call_rho = \
        batch_greeks_call(S, K, r, T, call_sigmas)

    put_delta, put_delta2, put_vega, put_gamma, put_theta, put_rho = \
        batch_greeks_put(S, K, r, T, put_sigmas)

    # Convert back to NumPy arrays for compatibility
    return [
        np.array(call_IVs),
        np.array(call_delta),
        np.array(call_delta2),
        np.array(call_vega),
        np.array(call_gamma),
        np.array(call_theta),
        np.array(call_rho),
        np.array(put_IVs),
        np.array(put_delta),
        np.array(put_delta2),
        np.array(put_vega),
        np.array(put_gamma),
        np.array(put_theta),
        np.array(put_rho),
    ]


# ============================================================================
# Additional Utility Functions
# ============================================================================

@jit
def put_call_parity_check(S, K, r, T, call_price, put_price):
    """
    Verify put-call parity: C - P = S - K*e^(-rT)

    Args:
        S: Underlying price
        K: Strike price
        r: Risk-free rate
        T: Time to expiration
        call_price: Call option price
        put_price: Put option price

    Returns:
        Parity difference (should be near 0 for European options)
    """
    pv_strike = K * jnp.exp(-r * T)
    return call_price - put_price - (S - pv_strike)


# Vectorize parity check
batch_parity_check = vmap(put_call_parity_check, in_axes=(0, 0, None, None, 0, 0))
