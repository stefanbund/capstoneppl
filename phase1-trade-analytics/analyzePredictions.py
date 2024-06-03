import os
import tempfile
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
test_path = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/PRED/test-miniature'
pred_path = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/PRED'
history_path = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/PREDICTION_HISTORY'
temp_path = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/PRED/TEMP'
folder_path = pred_path
n_days = 5 #how many days
# x_profit = 1.01 #percentage alpha you are seeking to settle for

def mean_time(times):
    # Convert all time strings to timedelta objects
    time_deltas = [timedelta(hours=int(t.split(':')[0]), 
                             minutes=int(t.split(':')[1]), 
                             seconds=int(t.split(':')[2])) for t in times]
    
    # Calculate the mean as a sum of all time deltas divided by the number of times
    mean_delta = sum(time_deltas, timedelta()) / len(time_deltas)
    
    # Convert the mean timedelta to a time format (HH:MM:SS)
    mean_time = (str(int(mean_delta.total_seconds() // 3600)).zfill(2) + ':' +
                 str(int((mean_delta.total_seconds() % 3600) // 60)).zfill(2) + ':' +
                 str(int(mean_delta.total_seconds() % 60)).zfill(2))
    
    return mean_time

def process_files():
    temp_dir = tempfile.mkdtemp(prefix='TEMP')

    print(f"Temporary folder created: {temp_dir}")
    import shutil

    pred_folder = pred_path  # Replace with the actual path to your 'PRED' folder
    history_folder = history_path  # Replace with the actual path to your 'HISTORY' folder

    # Copy files from 'PRED' folder
    for filename in os.listdir(pred_folder):
        src_file = os.path.join(pred_folder, filename)
        dst_file = os.path.join(temp_dir, filename)
        shutil.copy(src_file, dst_file)

    # Copy files from 'HISTORY' folder
    for filename in os.listdir(history_folder):
        src_file = os.path.join(history_folder, filename)
        dst_file = os.path.join(temp_dir, filename)
        shutil.copy(src_file, dst_file)

    print("Files copied successfully.")
    files = os.listdir(temp_dir)      # List all files in the given folder
    
    diffs = [] #trades which did not complete within the threshold metric, n days
    waits = [] #successful completions, as a count of rows, where 1 row = 1 minute
    for file_name in files:
        file_path = os.path.join(temp_dir, file_name)
        print(f"file path: {file_path}")
        df = pd.read_csv(file_path, header=None, names=['symbol', 'timestamp', 'method', 'y_pred'])
        row = df.iloc[0]
        print(f"row: {row}") #first row of df
        if row['symbol'] != 0: #trips up with the symbol '00-USD', 
            symbol = row['symbol'] + '-USD'
            timestamp = row['timestamp']
            start_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') #- timedelta(days=1)
            end_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') + timedelta(days=n_days)
            print(f'start time: {start_date},  end time: {end_date}')
            data = yf.download(symbol, start=start_date, end=end_date, interval='1m')#, exchange='coinbase')
            if(data.shape[0] >0): #length of data is something
                new_df = data[data.index >= timestamp]
                first_close_value = new_df['Close'].iloc[0]
                target_close_value = first_close_value * 1.01
                target_rows = new_df[new_df['Close'] >= target_close_value]
                if not target_rows.empty:
                    target_row_index = target_rows.index[0]
                    row_count = target_row_index - new_df.index[0]
                    analysis = {"timedelta":{row_count}, "symbol": {symbol}, "commence":{start_date}}
                    print(analysis)
                    waits.append(analysis)
                else:
                    row_count = target_row_index - new_df.index[0]  # Or set to None or another appropriate value
                    analysis_diff = {"timedelta":{row_count}, "symbol": {symbol}, "commence":{start_date}}
                    print(analysis_diff)
                    diffs.append(analysis_diff) #how many 
            else:
                print(f'no downloadable price data for symbol {symbol}')

        if row['symbol'] == 0: #trips up with the symbol '00-USD'
            symbol = '00-USD' #row['symbol'] + '0' +'-USD'
            timestamp = row['timestamp']
            start_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') #- timedelta(days=1)
            end_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') + timedelta(days=n_days)
            print(f'start time: {start_date},  end time: {end_date}')
            data = yf.download(symbol, start=start_date, end=end_date, interval='1m')#, exchange='coinbase')
            if(data.shape[0] >0): #length of data is something
                new_df = data[data.index >= timestamp]
                first_close_value = new_df['Close'].iloc[0]
                target_close_value = first_close_value * 1.01
                target_rows = new_df[new_df['Close'] >= target_close_value]
                if not target_rows.empty:
                    target_row_index = target_rows.index[0]
                    row_count = target_row_index - new_df.index[0]
                    analysis = {"timedelta":{row_count}, "symbol": {symbol}, "commence":{start_date}}
                    print(analysis)
                    waits.append(analysis)
                else:
                    row_count = target_row_index - new_df.index[0]  # Or set to None or another appropriate value
                    analysis_diff = {"timedelta":{row_count}, "symbol": {symbol}, "commence":{start_date}}
                    print(analysis_diff)
                    diffs.append(analysis_diff) #how many 
            else:
                print(f'no downloadable price data for symbol {symbol}')
            # print("Number of rows between the first row and the row where 'Close' is >= 1% more:", row_count)
        print("                                ")
    return diffs, waits

# Assuming the folder 'PRED' exists in the current directory
incompletes = process_files()
total_trades = len(incompletes[0]) + len(incompletes[1])
ratio_incomplete = (len(incompletes[1]) / total_trades) * 100
print(f'percentage completed within threshold, {n_days}: {ratio_incomplete}%')
# print(f"number of analyzed, incomplete trades, after 5 days: {len(incompletes[1])}")
waits = incompletes[1]
waits_meta = pd.DataFrame(waits)
print(waits_meta.columns)
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create the filename with the desired format
filename = f"waits-meta_{current_datetime}.csv"

# Save the DataFrame to the CSV file
waits_meta.to_csv(filename, index=False)

print(f"DataFrame saved to {filename}")
