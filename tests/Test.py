from OptionGreeksGPU.GreeksGPU import calculate_option_metrics
import pandas as pd
import numpy as np
import time
from datetime import datetime
from datetime import time as t
start_time = time.time()

# ---------------------------------------------
# Example 2 - Time in warmup and subsequent runs
# ---------------------------------------------
df = pd.read_csv('OpGreeksTestInput.csv', parse_dates=['expiry', 'DT'])
df = df[['name', 'expiry', 'strike', 'last_price_CE', 'last_price_PE', 'last_price_Und', 'GreekRef_CE', 'GreekRef_PE', 'DT']]
df.loc[df[f'last_price_CE'] == 0, f'last_price_CE'] = 0.0001
df.loc[df[f'last_price_PE'] == 0, f'last_price_PE'] = 0.0001
df.reset_index(inplace=True)

count =0
while True:
    start_time = time.time()

    interestRate = 5

    for i, [expiry, DT] in df[['expiry', 'DT']].drop_duplicates().iterrows():
        expiryDT = datetime.combine(expiry, t(15,30))
        daysToExpiration = (expiryDT - DT).total_seconds() / (24 * 3600)

        df = df[(df['expiry'] == expiry) & (df['DT'] == DT)]
        df.reset_index(drop=True, inplace=True)
        df_filtered = df.drop(['name', 'expiry', 'DT'], axis=1)
        df_filtered = df_filtered[['strike', 'last_price_Und', 'last_price_CE', 'GreekRef_CE', 'last_price_PE', 'GreekRef_PE']]
        optionData = df_filtered.to_numpy()
        print(optionData)

        Data = calculate_option_metrics(option_data=optionData,
                                        days_to_expiry=daysToExpiration,
                                        interest_rate=interestRate)
        Result_DF = pd.DataFrame(np.column_stack(Data), columns=['call_IVs', 'call_deltas', 'call_delta2s', 'call_vegas', 'call_gammas',
                                                'call_thetas', 'call_rhos', 'put_IVs', 'put_deltas', 'put_delta2s', 'put_vegas',
                                                'put_gammas', 'put_thetas', 'put_rhos'])

        result_df = pd.concat([df, Result_DF], axis=1)
        result_df.to_csv('OpGreeksTestOutput.csv')
    print(f"Greeks computation for all contracts took {(time.time() - start_time)} seconds")
    count += 1
    if count > 10:
        break