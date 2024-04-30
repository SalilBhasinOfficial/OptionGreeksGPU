import os
import importlib

cuda_available = 'CUDA_PATH' in os.environ
if cuda_available:
    greeks_module = importlib.import_module("OptionGreeksGPU.GreeksGPU")
else:
    greeks_module = importlib.import_module("OptionGreeksGPU.GreeksC")

calculate_option_metrics = getattr(greeks_module, 'calculate_option_metrics')