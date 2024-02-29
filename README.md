# OptionGreeksGPU

OptionGreeksGPU is a high-performance, GPU-accelerated library designed to calculate option Greeks for a large number of contracts quickly and efficiently. By leveraging the power of modern GPU architectures, OptionGreeksGPU can compute Greeks for 1648 option contracts in just 0.20 seconds (after warmup), offering a significant performance improvement over traditional CPU-based methods.

Features

	•	Fast Computation: Utilizes GPU acceleration to dramatically reduce computation times for option Greeks.
	•	Easy Integration: Designed to easily integrate with existing Python financial analysis workflows.
	•	Comprehensive: Supports a wide range of Greeks calculations, including Delta, Gamma, Theta, Vega, and Rho.

# Black-Scholes Model

The Black-Scholes model is a fundamental concept in modern financial theory and is widely used for pricing European options on stocks that do not pay dividends. Developed by Fischer Black, Myron Scholes, and Robert Merton in the early 1970s, this model provides a theoretical estimate of the price of European-style options. The beauty of the Black-Scholes model lies in its ability to factor in the major variables affecting option prices, such as the stock price, the exercise price, the risk-free interest rate, the time to expiration, and the volatility of the stock.

The Black-Scholes formula helps investors and traders to determine the fair value of an option, allowing them to make informed trading decisions. Specifically, it calculates the price of an option by inputting the current stock price, the option's strike price, the time until expiration (expressed as a year fraction), the risk-free interest rate, and the volatility of the stock. The model assumes that stock prices follow a lognormal distribution because asset prices cannot be negative, and it considers the constant risk-free rate for the option's life span.

The model is particularly suited for European options, which can only be exercised at expiration, unlike American options, which can be exercised at any time before or at expiration. It's important to note that while the Black-Scholes model provides a robust framework for option valuation, it does have limitations, especially when applied to American options, options on dividend-paying stocks, or in highly volatile markets.


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

### Input Format for OptionGreeksGPU
When using the OptionGreeksGPU library to calculate option Greeks based on the Black-Scholes model, the input data should be structured as follows:

input_data = (
    option_data=[[strikePrices], [underlyingPrices], [callPrices], [callRefs = 0s], [putPrices], [putRefs = 1s]],
    days_to_expiry,
interest_rate
)

strikePrices: An array of strike prices for the options.
underlyingPrices: An array of current prices of the underlying asset.
callPrices: An array of market prices for call options.
callRefs: An array filled with 0s, indicating that the corresponding prices are for call options.
putPrices: An array of market prices for put options.
putRefs: An array filled with 1s, indicating that the corresponding prices are for put options.
interestRate: The risk-free interest rate, expressed as a whole number (e.g., 5 for 5%).
daysToExpiry: The time to expiration of the options, expressed in days (with Decimals).


# Performance

	•	CPU (multiprocessing): ~8 minutes for 1648 option contracts. (using mibian library in multiprocessing 12 cores)
	•	GPU (OptionGreeksGPU): 0.20 seconds for 1648 option contracts (after warmup).

# Contributing

Contributions are welcome! If you’d like to contribute, please fork the repository, create a feature branch, and submit a pull request.

# License

OptionGreeksGPU is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.

# Support

If you encounter any problems or have any suggestions, please open an issue on the project’s GitHub page.
