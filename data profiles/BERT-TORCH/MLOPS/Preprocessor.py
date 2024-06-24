
import pandas as pd
import os
import numpy as np
import requests #get all symbols from coinbase via query

# API endpoint for cryptocurrencies
url = "https://api.coinbase.com/v2/currencies/crypto"

# Make a GET request to retrieve the list of cryptocurrencies
response = requests.get(url).json()

# Extract the relevant information (e.g., code and name)
cryptos = [(coin["code"], coin["name"]) for coin in response["data"]]

# Print the list of cryptocurrencies
for pair, name in cryptos:
    # print(f"{code}: {name}")
    form = pair + "-USD"  #AVAX-USD
    print(form)
    ret = [] #FORMERLY WALKING THE LOB_CAPS FOLDER, now loads all cell2024-x.csv files
    # Step 1: List files starting with "cell"
    directory = "/home/stefan/Desktop/GRUS-CSV-SAMPLER-DATA/"  #os.getcwd()  # Get current directory
    file_list = [filename for filename in os.listdir(directory) if filename.startswith("cell")]
    # Initialize master list
    master = []
    # Step 2: Load each file as a Pandas DataFrame
    for filename in file_list:
        filepath = os.path.join(directory, filename)
        try:
            provisional = pd.read_csv(filepath,index_col=None, header=0)  # Assuming CSV files, adjust as needed
            print(form, provisional.columns)
            # Step 3: Search for rows where "symbol" matches the value of "pair"
            symbol_rows = provisional[provisional["symbol"] == form]
            # Step 4: Append matching rows to master list
            master.append(symbol_rows)
            # master.extend(symbol_rows.to_dict(orient="records"))
        except pd.errors.EmptyDataError:
            # Handle empty files (if needed)
            pass
    capsFrame = pd.concat(master, axis=0, ignore_index=True) #end frame contains all data
    capsFrame.drop('symbol', axis=1, inplace=True)

    capsFrame.sort_values(by=['time'], ascending=True)   #sorted by time into one time series
    print("for new df: ", capsFrame.shape[0])
    if capsFrame.shape[0] > 0:
        start = capsFrame["time"].min()
        end = capsFrame["time"].max()
        print("start: ", start, " end: ", end)
        print(capsFrame.columns)

        # impute missing values with last non-null value
        capsFrame['bc'] = capsFrame['bc'].fillna(method='ffill')
        capsFrame['ac'] = capsFrame['ac'].fillna(method='ffill')
        capsFrame['tbv'] = capsFrame['tbv'].fillna(method='ffill')
        capsFrame['tav'] = capsFrame['tav'].fillna(method='ffill')
        capsFrame['mp'] = capsFrame['mp'].fillna(method='ffill')
        # capsFrame['minBid'] = capsFrame['minBid'].fillna(method='ffill')

        # Load your time series data into a pandas dataframe
        # consider cahnging this approach because it doesnt actually check in between values
        caps_df = capsFrame   
        lookback_period = 10 # in rows
        caps_df['change'] = caps_df['mp'].pct_change(periods=lookback_period)
        caps_df['bc_change'] = caps_df['bc'].pct_change(periods=lookback_period)
        caps_df['ac_change'] = caps_df['ac'].pct_change(periods=lookback_period)
        caps_df['tav_change'] = caps_df['tav'].pct_change(periods=lookback_period)
        caps_df['tbv_change'] = caps_df['tbv'].pct_change(periods=lookback_period)
        ## key components: bc_change, ac_change, tav_change, tbv_change, change
        print(caps_df.shape[0], caps_df.columns)# Calculate the returns of your asset over a fixed lookback period

        meanChange = round(caps_df['change'].mean(),8)

        # identify units of 10 rows where the percent change is greater or less than the threshold
        ### key components: bc_change, ac_change, tav_change, tbv_change, change
        threshold = meanChange
        surges = []
        precursors = []
        for i in range(0,len(caps_df),10):
            if caps_df.iloc[i:i+10]['change'].mean() >= threshold:
                surges.append({'time': caps_df.iloc[i]['time'],
                            's_MP': caps_df.iloc[i]['mp'],
                            'change': caps_df.iloc[i:i+10]['change'].mean(),
                            'type':'surge'})  #['bc', 'ac', 'tbv', 'tav', 'time', 'mp', 'minBid', 'change']
            else:
                precursors.append({'time': caps_df.iloc[i]['time'],
                                'p_MP': caps_df.iloc[i]['mp'],
                                'change': caps_df.iloc[i:i+10]['change'].mean(),
                                    'type':'precursor',
                                    'precursor_buy_cap_pct_change':caps_df.iloc[i]['bc_change'], 
                                    'precursor_ask_cap_pct_change':caps_df.iloc[i]['ac_change'],
                                    'precursor_bid_vol_pct_change':caps_df.iloc[i]['tbv_change'],
                                    'precursor_ask_vol_pct_change':caps_df.iloc[i]['tav_change']
                                    })  

        surges_df = pd.DataFrame(surges)
        precursors_df = pd.DataFrame(precursors)
        sequence_df = pd.concat([surges_df, precursors_df]).sort_values(by=['time'], ascending=True) #bigger values at the end of the list

        ### data mining 2: information gain, create new features
        sequence_df['group'] = (sequence_df['type'] != sequence_df['type'].shift(1)).cumsum()
        columns_to_transform = [
            'precursor_buy_cap_pct_change',
            'precursor_ask_cap_pct_change',
            'precursor_bid_vol_pct_change',
            'precursor_ask_vol_pct_change'
        ]
        for col in columns_to_transform:
            sequence_df[col] = sequence_df.groupby('group')[col].transform(lambda x: x.sum() if not x.isna().all() else np.nan)

        # # impute missing values with last non-null value DONE PRIOR, NOW AT START
        sequence_df['s_MP'] = sequence_df['s_MP'].fillna(method='ffill')
        sequence_df['p_MP'] = sequence_df['p_MP'].fillna(method='ffill')
        sequence_df['precursor_buy_cap_pct_change'] = sequence_df['precursor_buy_cap_pct_change'].fillna(method='ffill')
        sequence_df['precursor_ask_cap_pct_change'] = sequence_df['precursor_ask_cap_pct_change'].fillna(method='ffill')
        sequence_df['precursor_bid_vol_pct_change'] = sequence_df['precursor_bid_vol_pct_change'].fillna(method='ffill')
        sequence_df['precursor_ask_vol_pct_change'] = sequence_df['precursor_ask_vol_pct_change'].fillna(method='ffill')

        ### critical grouped statistics
        sequence_df['length'] = sequence_df.groupby(['type', 'group'])['group'].transform('count')

        print(sequence_df.shape[0])
        sequence_df['sum_change'] = sequence_df.groupby(['type', 'group'])['change'].transform('sum')

        sequence_df['max_surge_mp'] = sequence_df.groupby(['type', 'group'])['s_MP'].transform('max')
        sequence_df['min_surge_mp'] = sequence_df.groupby(['type', 'group'])['s_MP'].transform('min')

        sequence_df['max_precursor_mp'] = sequence_df.groupby(['type', 'group'])['p_MP'].transform('max')
        sequence_df['min_precursor_mp'] = sequence_df.groupby(['type', 'group'])['p_MP'].transform('min')

        sequence_df['area']  = sequence_df.apply(lambda row: row['length'] * row['sum_change'], axis=1)

        sequence_df.loc[sequence_df['type'] == 'surge', 'surge_area'] = sequence_df.loc[sequence_df['type'] == 'surge', 'area']

        sequence_df['surge_targets_met_pct']  = sequence_df.apply(lambda group: ((group['max_precursor_mp']-group['max_surge_mp'])/group['max_surge_mp']  ) *100, axis=1)

        ## data mining 3: form final sequences by statistical weight
        # Critical group by unique identifier
        unique_df = sequence_df.groupby('group').first().reset_index()


        # merge even and odd rows: needs to start with a precursor removes the first surge
        unique_df = unique_df.iloc[1:]
        even_df = unique_df.iloc[::2].reset_index(drop=True)
        odd_df = unique_df.iloc[1::2].reset_index(drop=True)

        merged_df = pd.concat([even_df, odd_df], axis=1)


        #final df:
        final_df = merged_df.dropna(axis=1, how='all')
        final_df.columns= list(final_df.columns[:-1]) + ['surge_targets_met_pct.1']
        # print("length of final DF: ", final_df.shape[0], final_df.columns)
        bins = [
        # final_df['surge_targets_met_pct'].min() - 1,
        -4/3,
        # -4/6,
        # -4/12,
        # 0,
        0.125,
        0.25,
        0.5,
        0.75,
        1,
        2,
        # final_df['surge_targets_met_pct'].max() + 1
        ]
        bin_labels = list(range(1, len(bins)))
        final_df['label'] = pd.cut(final_df['surge_targets_met_pct'], bins=bins, labels=bin_labels)

        final_df_binary = final_df.iloc[:, :-1]
        final_df_binary['label'] = (final_df_binary['surge_targets_met_pct']> 0.74).astype(int)
        final_df_binary.to_csv(f'./BBP/{pair}-binary_binned_pipeline.csv')


# def getCAPSByDateAndType(pair):  #returns a dict, date + df caps for that date, then extended date and time
#                                     # print("for type, ", type)  ./lob_caps/
#         ret = [] #FORMERLY WALKING THE LOB_CAPS FOLDER, now loads all cell2024-x.csv files
#         # Step 1: List files starting with "cell"
#         directory = os.getcwd()  # Get current directory
#         file_list = [filename for filename in os.listdir(directory) if filename.startswith("cell")]
#         # Initialize master list
#         master = []
#         # Step 2: Load each file as a Pandas DataFrame
#         for filename in file_list:
#             filepath = os.path.join(directory, filename)
#             try:
#                 provisional = pd.read_csv(filepath)  # Assuming CSV files, adjust as needed
#                 print(provisional.columns)
#                 # Step 3: Search for rows where "symbol" matches the value of "pair"
#                 symbol_rows = provisional[provisional["symbol"] == self.pair]
#                 # Step 4: Append matching rows to master list
#                 master.extend(symbol_rows.to_dict(orient="records"))
#             except pd.errors.EmptyDataError:
#                 # Handle empty files (if needed)
#                 pass
#         return master # a list of rows matching self.pair
        
        