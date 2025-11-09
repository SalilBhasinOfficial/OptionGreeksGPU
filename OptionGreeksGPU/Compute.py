"""
Compute.py - Backend selection for Option Greeks computation

This module automatically selects the best available backend:
1. JAX (preferred) - Automatic differentiation, unified CPU/GPU/TPU code
2. Numba CUDA - For GPU acceleration when JAX is not available
3. Numba JIT - CPU fallback with machine code compilation

The selection is transparent to the user via the calculate_option_metrics API.
"""

import subprocess
import importlib
import sys


def is_jax_available():
    """Check if JAX library is available."""
    try:
        import jax
        return True
    except ImportError:
        return False


def is_cuda_available():
    """Check if CUDA-compatible GPU is available."""
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False


# Determine the best available backend
jax_available = is_jax_available()
cuda_available = is_cuda_available()

# Backend selection priority: JAX > CUDA > CPU
if jax_available:
    backend_name = "JAX"
    greeks_module = importlib.import_module("OptionGreeksGPU.GreeksJAX")
    print(f"OptionGreeksGPU: Using JAX backend (automatic differentiation enabled)")
    print(f"  JAX devices available: ", end="")
    try:
        import jax
        devices = jax.devices()
        print(f"{[str(d) for d in devices]}")
    except:
        print("[unknown]")
elif cuda_available:
    backend_name = "CUDA"
    greeks_module = importlib.import_module("OptionGreeksGPU.GreeksGPU")
    print(f"OptionGreeksGPU: Using Numba CUDA backend (GPU acceleration)")
else:
    backend_name = "CPU"
    greeks_module = importlib.import_module("OptionGreeksGPU.GreeksC")
    print(f"OptionGreeksGPU: Using Numba JIT backend (CPU machine code)")

# Export the main calculation function
calculate_option_metrics = getattr(greeks_module, 'calculate_option_metrics')

# Export backend information
__backend__ = backend_name