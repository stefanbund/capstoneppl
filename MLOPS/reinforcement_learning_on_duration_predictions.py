#COPYRIGHT STEFAN L. BUND, copyright available at https://github.com/stefanbund/radidisco. No use is authorized and no sharing license is granted.
'''PREMISE: jULY 11 - get all finished uration based predictions in SCP-PRED. These are based upon a fitted duration model. 
get all finished predictions, determine their time to finish. Store them on a per-symbol basis, after generating a per-prediction assessment.
1. get all scp pred
2. use yf to estimate time duration, dump to prediction_duration_folder
3. aggregate per symbol in aggregated_durations_by_symbol
ready to viz
scp-pred file format:
2024-07-11 05:01:48,1,KNeighborsClassifier(),-0.011209928740358643,6.156138932289501e-06,-0.0018153137746962367,-0.0002183443696004428,-0.002100452479913417,4,1
'''
import os
import tempfile
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import shutil

pred_path = '/home/stefan/Desktop/STADIUM-DATA/SCP-PRED' #SCP-PRED '/home/stefan/Desktop/STADIUM-DATA/DIRECTIONAL_PREDICTIONS' 
n_days = 5 #how many days
prediction_duration_folder = '/home/stefan/Desktop/STADIUM-DATA/ALL_END_PREDICTIONS_TIME_DURATIONS/'#'/home/stefan/Desktop/STADIUM-DATA/PREDICTION_DURATION/'  #changes: line 39, 73, replaces WM-BUILD
aggregated_durations_by_symbol = '/home/stefan/Desktop/STADIUM-DATA/END_RESULT_EFFICIENCY/' #/home/stefan/Desktop/STADIUM-DATA/SYMBOL_DURATIONS_FITTABLE/' #TESTLIST successor, 


def buildUniquesWAITMETA():  #supplies all unique symbols in the waits-meta folder, then a list of all files, will match below
    file_names = os.listdir(f"{prediction_duration_folder}")  # List all files in the folder, for the duration analyses, was {prefix}WM-BUILD
    unique_prefixes = set()  # Extract unique prefixes before the '-' character
    all_filenames = []
    for name in file_names:  #iterate all files, separating first characters as symbol
        prefix_h = name.split('-')[0]
        unique_prefixes.add(prefix_h)
        all_filenames.append(name)
    return unique_prefixes, all_filenames   #all symbols, then all named files, will iterate both

def make_duration_dataset():
    names_list = buildUniquesWAITMETA() # the list of all symbols processed here today
    all_files_list = names_list[1]   #the files there, many thousands, one per trade prediction
    all_symbol_uniques = names_list[0]  #the symbols as a set or uniques, for which there may be many predictions, unlimited
    print(f"all files: \n {all_files_list}")  #file names, not absolute pathed
    print(f"all names: \n {all_symbol_uniques}")

    for name in all_symbol_uniques:
        print(f" for symbol, {name}") #prove symbol name
        symbol_wm_df = pd.DataFrame()

        for file_name in all_files_list:
            symbol_front = file_name.split('-')[0]  # Extract the part before the hyphen
            print(f"for {symbol_front}, file {file_name}")
            if symbol_front == name: #file contains the unique, lump together
                wm_file = f"{prediction_duration_folder}{file_name}"
                wmdf = pd.read_csv(wm_file) 
                wmdf = wmdf.dropna(subset=['hours'])
                wmdf.loc[:, 'bin'] = wmdf['hours'].apply(lambda x: 0 if x > 3.0 else 1)
                # single_df = pd.read_csv(f"{prefix2}{file_name}")
                symbol_wm_df = pd.concat([symbol_wm_df, wmdf], ignore_index=True)
                #use variable, symbol_aggregation_durations
        symbol_wm_df.to_csv(f"{aggregated_durations_by_symbol}{name}-USD-waits-meta-data.csv")  #was {prefix}/TESTLIST/
    return
        
def clean_and_convert( date_str):
    print(f"OPEN CLEAN AND CONVERT, {type(date_str)}, with date string: {date_str}")
    try:
        cleaned_data = date_str.strip("{}").replace('datetime.datetime', '') # Remove the curly braces and 'datetime.datetime'
        print(f"cleaned data: {cleaned_data}")
        numbers = cleaned_data.strip("()").split(", ") # Remove the parentheses and split the string
        while len(numbers) < 6:  # Ensure there are exactly six components
            numbers.append("0")
        formatted_data = f"({', '.join(numbers)})" # Construct the final formatted string
        return datetime.strptime(formatted_data, "(%Y, %m, %d, %H, %M, %S)") # Convert to a datetime object
    except Exception as e:
        print(f"clean and covert issue, {e}")

def mean_time(times):  # Convert all time strings to timedelta objects
    time_deltas = [timedelta(hours=int(t.split(':')[0]), 
                             minutes=int(t.split(':')[1]), 
                             seconds=int(t.split(':')[2])) for t in times]
    mean_delta = sum(time_deltas, timedelta()) / len(time_deltas)
    mean_time = (str(int(mean_delta.total_seconds() // 3600)).zfill(2) + ':' +
                 str(int((mean_delta.total_seconds() % 3600) // 60)).zfill(2) + ':' +
                 str(int(mean_delta.total_seconds() % 60)).zfill(2))
    return mean_time

def init(): #set up the analysis in the TEMP folder, copy to temp foler in /tmp
    print("enter init")
    temp_dir = tempfile.mkdtemp(prefix='TEMP')
    print(f"Temporary folder created: {temp_dir}")
    pred_folder = pred_path  # Replace with the actual path to your 'PRED' folder
    for filename in os.listdir(pred_folder):  # Copy files from 'PRED' folder
        src_file = os.path.join(pred_folder, filename)
        dst_file = os.path.join(temp_dir, filename)
        shutil.copy(src_file, dst_file)
    return temp_dir #where I move all the experiment's files, to get started, I return so i can iterate
    

def process_file(file_name, symbol, tmp): #send it the name of the prediction to study, as stored in /tmp/TEMP*
    
    print(f"process file {file_name} for symbol, {symbol}")

    analysis = {} #populate below
    file_path = os.path.join(tmp, file_name)
    print(f"accessing prediction at: {file_path}")
    #[self.symbol, time, exp, y_pred, buy_cap, ask_cap, bid_vol, ask_vol, sum_change, length]
    #2024-07-11 05:01:48,1,KNeighborsClassifier(),-0.011209928740358643,6.156138932289501e-06,-0.0018153137746962367,-0.0002183443696004428,-0.002100452479913417,4,1
    # data = ['time', 'y_pred', 'model_meta', 'precursor_buy_cap_pct_change', 'precursor_ask_cap_pct_change','precursor_bid_vol_pct_change',  
    #                 'precursor_ask_vol_pct_change','sum_change','length', 'duration_y_pred'] 
    df = pd.read_csv(file_path, header=None, names=['timestamp', 'y_pred', 'method', 'precursor_buy_cap_pct_change', 'precursor_ask_cap_pct_change','precursor_bid_vol_pct_change',  
                    'precursor_ask_vol_pct_change','sum_change','length', 'duration_y_pred'])
    if df.empty:
        print(f" df for symbol, {file_path} is empty")
        return
    else:
        row = df.iloc[0]
        print(f"prediction content: {row}") #first row of df
        waits =[]
        if symbol != '00': #trips up with the symbol '00-USD', row['symbol']
            symbol = symbol + '-USD'
            timestamp = row['timestamp']
            start_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') #- timedelta(days=1)
            end_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') + timedelta(days=n_days)
            print(f'for YF, start time: {start_date},  end time: {end_date}')
            data = yf.download(symbol, start=start_date, end=end_date, interval='1m')#, exchange='coinbase')
            if(data.shape[0] >0): #length of data is something
                new_df = data[data.index >= timestamp]
                first_close_value = new_df['Close'].iloc[0]
                target_close_value = first_close_value * 1.01
                target_rows = new_df[new_df['Close'] >= target_close_value]
                if not target_rows.empty:
                    target_row_index = target_rows.index[0]
                    row_count = target_row_index - new_df.index[0]
                    
                    analysis = {"timedelta":str(row_count), "symbol":symbol, "commence":start_date,
                                'timestamp':df['timestamp'].iloc[0], 'method':df['method'].iloc[0], 'y_pred':df['y_pred'].iloc[0],
                            'precursor_buy_cap_pct_change':df['precursor_buy_cap_pct_change'].iloc[0], 
                            'precursor_ask_cap_pct_change':df['precursor_ask_cap_pct_change'].iloc[0], 
                            'precursor_bid_vol_pct_change':df['precursor_bid_vol_pct_change'].iloc[0], 
                            'precursor_ask_vol_pct_change':df['precursor_ask_vol_pct_change'].iloc[0], 
                            'sum_change':df['sum_change'].iloc[0], 'length':df['length'].iloc[0]}
                    print(f"analysis of prediction: {analysis}")
                    waits.append(analysis)
            else:
                print(f'no downloadable price data for symbol {symbol}')

        if symbol == '00': #trips up with the symbol '00-USD'
            symbol = '00-USD' #row['symbol'] + '0' +'-USD'  alter, make adjustable to input
            timestamp = row['timestamp']
            start_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') #- timedelta(days=1)
            end_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') + timedelta(days=n_days)
            data = yf.download(symbol, start=start_date, end=end_date, interval='1m')#, exchange='coinbase')
            if(data.shape[0] >0): #length of data is something
                new_df = data[data.index >= timestamp]
                first_close_value = new_df['Close'].iloc[0]
                target_close_value = first_close_value * 1.01
                target_rows = new_df[new_df['Close'] >= target_close_value]
                if not target_rows.empty:
                    target_row_index = target_rows.index[0]
                    row_count = target_row_index - new_df.index[0]
                    analysis = {"timedelta":str(row_count), "symbol":symbol, "commence":start_date,
                                'timestamp':df['timestamp'].iloc[0], 'method':df['method'].iloc[0], 'y_pred':df['y_pred'].iloc[0],
                            'precursor_buy_cap_pct_change':df['precursor_buy_cap_pct_change'].iloc[0], 
                            'precursor_ask_cap_pct_change':df['precursor_ask_cap_pct_change'].iloc[0], 
                            'precursor_bid_vol_pct_change':df['precursor_bid_vol_pct_change'].iloc[0], 
                            'precursor_ask_vol_pct_change':df['precursor_ask_vol_pct_change'].iloc[0], 
                            'sum_change':df['sum_change'].iloc[0], 'length':df['length'].iloc[0]}
                    print(f"waits: {analysis}")
                    waits.append(analysis)
            else:
                print(f'no downloadable price data for symbol {symbol}')
        #TODO: change waits_meta to waits_df 
        waits_df = pd.DataFrame(waits) 
        try: #for now, reduce the 
            waits_df['timedelta'] = pd.to_timedelta(waits_df['timedelta'] ) #.str.strip('{}'))

            # Now you can safely use .total_seconds() to convert the timedelta to hours
            waits_df['hours'] = waits_df['timedelta'].dt.total_seconds() / 3600
            waits_df['commence'] = pd.to_datetime(waits_df['commence'])

            waits_df['commence'] = pd.to_datetime(waits_df['commence']) 

            waits_df['time'] = (waits_df['commence'] - pd.Timestamp('1970-01-01')).dt.total_seconds()  # Calc epoch sec
            waits_df['efficiency'] = waits_df['hours'].apply(lambda x: 1 if x < 3.0 else 0)  #hours becomes a label like qualifier
            
            combined_df = pd.concat([waits_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
            print(f"waits meta information 3: \n{combined_df.head(1)}") #ok

            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            waits_df.to_csv(f"{prediction_duration_folder}{symbol}-WAITS-META-{current_datetime}.csv")  #waits meta data is now complete for that trade
            print("                                ")
        except Exception as e:
            print(f"waits meta dataframe issue: {e}")
    return 

def run_analysis(experiment_tmp):  # Iterate through all the files in the '/tmp' folder
    print(f"run analysis starts. temp directory: {experiment_tmp}")
    for filename in os.listdir(experiment_tmp):
        print(f"1, get PRED,  filename, {filename}")
        # symbol = ''
        # filename = 'SPA-prediction-log-2024-07-10 20:18:25.csv'
        # Split the filename on the first occurrence of '-' and get the first part
        symbol = filename.split('-', 1)[0]
        print(f"start analysis for symbol, {symbol}")

        # try:
        # print(f"get symbol file, initiate analysis")
        if filename.endswith('.csv'):
            original_string = filename
            index_of_dash = original_string.find('-')
            if index_of_dash != -1:
                substring_before_dash = original_string[:index_of_dash]
            else:
                substring_before_dash = ""
            symbol = substring_before_dash
        file_path = os.path.join(experiment_tmp, filename)
        process_file(file_path, symbol, experiment_tmp) #push the duration analysis for one trade to WM-BUILD
    make_duration_dataset()

    return
    
run_analysis(init()) #determines a duration, per predicted trade in the PRED folder
# make_duration_dataset()  #as of june 22, predict stores precursor features, and we get duratoin, then save 2 csv
