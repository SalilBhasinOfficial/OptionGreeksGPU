import subprocess
import importlib
def is_cuda_available():
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False

cuda_available = is_cuda_available()
print("CUDA Available:", cuda_available)


if cuda_available:
    greeks_module = importlib.import_module("OptionGreeksGPU.GreeksGPU")
else:
    greeks_module = importlib.import_module("OptionGreeksGPU.GreeksC")

calculate_option_metrics = getattr(greeks_module, 'calculate_option_metrics')