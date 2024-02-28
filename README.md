# OptionGreeksGPU

OptionGreeksGPU is a high-performance, GPU-accelerated library designed to calculate option Greeks for a large number of contracts quickly and efficiently. By leveraging the power of modern GPU architectures, OptionGreeksGPU can compute Greeks for 1648 option contracts in just 0.20 seconds (after warmup), offering a significant performance improvement over traditional CPU-based methods.

Features

	•	Fast Computation: Utilizes GPU acceleration to dramatically reduce computation times for option Greeks.
	•	Easy Integration: Designed to easily integrate with existing Python financial analysis workflows.
	•	Comprehensive: Supports a wide range of Greeks calculations, including Delta, Gamma, Theta, Vega, and Rho.

# Installation

Before installing OptionGreeksGPU, ensure you have a CUDA-compatible GPU and the appropriate CUDA Toolkit installed on your system.

To install OptionGreeksGPU, run the following command:

pip install OptionGreeksGPU

# Usage

Here’s a quick example of how to use OptionGreeksGPU (see test.py for exact operation) to compute Greeks for option contracts:

from OptionGreeksGPU.GreeksGPU import calculate_option_metrics
import pandas as pd

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

# Performance

	•	CPU (multiprocessing): ~8 minutes for 1648 option contracts. (using mibian library in multiprocessing 12 cores)
	•	GPU (OptionGreeksGPU): 0.20 seconds for 1648 option contracts (after warmup).

# Contributing

Contributions are welcome! If you’d like to contribute, please fork the repository, create a feature branch, and submit a pull request.

# License

OptionGreeksGPU is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.

# Support

If you encounter any problems or have any suggestions, please open an issue on the project’s GitHub page.