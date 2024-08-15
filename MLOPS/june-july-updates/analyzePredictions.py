#PREMISE: determine the time duration required to render a trade completed. A dataframe is rendered on a per-symbol basis
# that associates precursor features with the predicted efficiency of the trade. This file works upon .csv files under PRED
# where a [1] class prediction was made. It queries yFinance to deliver a time horizon for each recommendation under PRED, 
# in a back-tested way. Each recommendation is evaluated with an efficiency metric, then a new WAITS-META file is produced
# per symbol. This new WAITS-META csv is read by the Predictor-WAITSMETA to evaluate whether the PRED recommendation
# is likely to settle within our desired time frame.
import os
import tempfile
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import subprocess #do scp when ready
import shutil

test_path = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/PRED/test-miniature'
pred_path = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/PRED'
history_path = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/PREDICTION_HISTORY'
temp_path = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/PRED/TEMP'
prefix = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/'
model_folder = "/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/MODEL" #use to get best model
folder_path = pred_path
n_days = 5 #how many days

def build_classifier(classifier_name, params): 
    if classifier_name == 'LogisticRegression':
        classifier = LogisticRegression(**params)
    if classifier_name == 'KNeighborsClassifier':
        classifier = KNeighborsClassifier(**params)
    if classifier_name == 'BernoulliNB':
        classifier = BernoulliNB(**params)
    return  classifier

def buildUniquesWAITMETA():  #supplies all unique symbols in the waits-meta folder, then a list of all files, will match below
    file_names = os.listdir(f"{prefix}WM-BUILD")  # List all files in the folder, for the duration analyses
    unique_prefixes = set()  # Extract unique prefixes before the '-' character
    all_filenames = []
    for name in file_names:  #iterate all files, separating first characters as symbol
        prefix_h = name.split('-')[0]
        unique_prefixes.add(prefix_h)
        all_filenames.append(name)
    return unique_prefixes, all_filenames   #all symbols, then all named files, will iterate both

def getBBP_symbol_file(symbol): #searches for symbol file in binary binned pipeline, returns a file url
    bbp_loc = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/BBP/'
    filename_b = f"{bbp_loc}{symbol}-binary_binned_pipeline.csv"
    print(f"getBBP_symbol_file, filename {filename_b}")
    return filename_b

def make_duration_dataset(): #runs after run_analysis()
    #iterate the waits meta folder, gather all symbol files into a single, symbol determined dataframe, then write to csv
    #the product of this is to be loaded when Predictor.py finds a [1] prediction, and must predict the duration, or 
    #wait, for the symbol's prediction. file location: {prefix}WM-BUILD/ASM-USD-waits-meta-data.csv
    names_list = buildUniquesWAITMETA() # the list of all symbols processed here today
    all_files_list = names_list[1]   #the files there, many thousands, one per trade prediction
    all_symbol_uniques = names_list[0]  #the symbols as a set or uniques, for which there may be many predictions, unlimited
    print(f"all files: \n {all_files_list}")
    print(f"all names: \n {all_symbol_uniques}")

    for name in all_symbol_uniques:
        print(f" for symbol, {name}")
        #get all files with that prefix:
        build_dir = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/WM-BUILD/' #from run_analysis step 2
        symbol_wm_df = pd.DataFrame()

        all_files_list = [file for file in os.listdir(build_dir) if file.startswith(name)]
        for file_name in all_files_list:
            # prefix2 = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/WM-BUILD/' #from run_analysis step 2
            wm_file = f"{build_dir}{file_name}"
            # filename = getBBP_symbol_file(name) #was symbol, in the notebook
            symboldf = pd.read_csv(getBBP_symbol_file(name))  #elim, as we simply take PRED file features (cap, vol etc)
            wmdf = pd.read_csv(wm_file) #duration data for that 

            # print(f"BBP df columns: \n{symboldf.columns}\n{wmdf.columns}")
            merged_df = symboldf.merge(wmdf, on='time', how='outer') #how to merge wait and BBP into one record
            merged_df = merged_df.dropna(subset=['hours'])
            merged_df.loc[:, 'bin'] = merged_df['hours'].apply(lambda x: 0 if x > 3.0 else 1)
            symbol_front = file_name.split('-')[0]  # Extract the part before the hyphen
            print(f"for {symbol_front}, file {file_name}")
            if symbol_front == name: #file contains the unique, lump together
                # single_df = pd.read_csv(f"{prefix2}{file_name}")
                symbol_wm_df = pd.concat([symbol_wm_df, merged_df], ignore_index=True) 
                symbol_wm_df.to_csv(f"{prefix}/TESTLIST/{name}-USD-waits-meta-data.csv") 

        # symbol_wm_df = pd.DataFrame()
        # for file_name in all_files_list:
        #     # prefix2 = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/WM-BUILD/' #from run_analysis step 2
        #     wm_file = f"{prefix2}{file_name}"
        #     # filename = getBBP_symbol_file(name) #was symbol, in the notebook
        #     symboldf = pd.read_csv(getBBP_symbol_file(name))
        #     wmdf = pd.read_csv(wm_file) #duration data for that 

        #     print(f"BBP df columns: \n{symboldf.columns}\n{wmdf.columns}")
        #     merged_df = symboldf.merge(wmdf, on='time', how='outer') #how to merge wait and BBP into one record
        #     merged_df = merged_df.dropna(subset=['hours'])
        #     merged_df.loc[:, 'bin'] = merged_df['hours'].apply(lambda x: 0 if x > 3.0 else 1)
        #     symbol_front = file_name.split('-')[0]  # Extract the part before the hyphen
        #     print(f"for {symbol_front}, file {file_name}")
        #     if symbol_front == name: #file contains the unique, lump together
        #         # single_df = pd.read_csv(f"{prefix2}{file_name}")
        #         symbol_wm_df = pd.concat([symbol_wm_df, merged_df], ignore_index=True) 
        #         symbol_wm_df.to_csv(f"{prefix}/TESTLIST/{name}-USD-waits-meta-data.csv") 
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
        # print(f"{datetime.strptime(formatted_data, "(%Y, %m, %d, %H, %M, %S)")}")
        return datetime.strptime(formatted_data, "(%Y, %m, %d, %H, %M, %S)") # Convert to a datetime object
    except Exception as e:
        print(f"clean and covert issue, {e}")

def getBBP_symbol_file(symbol): #searches for symbol file in binary binned pipeline, returns a file url
    bbp_loc = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/BBP/'
    filename_b = f"{bbp_loc}{symbol}-binary_binned_pipeline.csv"
    print(f"getBBP_symbo_file, filename {filename_b}")
    return filename_b

def mean_time(times):  # Convert all time strings to timedelta objects
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

def init(): #set up the analysis in the TEMP folder
    print("enter init")
    temp_dir = tempfile.mkdtemp(prefix='TEMP')
    print(f"Temporary folder created: {temp_dir}")
    pred_folder = pred_path  # Replace with the actual path to your 'PRED' folder
    history_folder = history_path  # Replace with the actual path to your 'HISTORY' folder        
    for filename in os.listdir(pred_folder):  # Copy files from 'PRED' folder
        src_file = os.path.join(pred_folder, filename)
        dst_file = os.path.join(temp_dir, filename)
        shutil.copy(src_file, dst_file)
    for filename in os.listdir(history_folder): # Copy files from 'HISTORY' folder
        src_file = os.path.join(history_folder, filename)
        dst_file = os.path.join(temp_dir, filename)
        shutil.copy(src_file, dst_file)
    return temp_dir #where I move all the experiment's files, to get started, I return so i can iterate
    

def process_file(file_name, symbol, tmp): #send it the name of the prediction to study, as stored in /tmp/TEMP*
    # files = os.listdir(tmp)      # List all files in the TEMP folder
    # for file_name in files:
    print(f"process file {file_name} for symbo, {symbol}")

    # diffs = [] #trades which did not complete within the threshold metric, n days
    # waits = [] #successful completions, as a count of rows, where 1 row = 1 minute
    analysis = {} #populate below
    file_path = os.path.join(tmp, file_name)
    print(f"accessing prediction at: {file_path}")
    #[self.symbol, time, exp, y_pred, buy_cap, ask_cap, bid_vol, ask_vol, sum_change, length]
    df = pd.read_csv(file_path, header=None, names=['symbol', 'timestamp', 'method', 'y_pred',
                            'buy_cap', 'ask_cap', 'bid_vol', 'ask_vol', 'sum_change', 'length'])
    if df.empty:
        print(f" df for symbol, {file_path} is empty")
        return
    else:
        row = df.iloc[0]
        print(f"prediction content: {row}") #first row of df
        waits =[]
        if row['symbol'] != 0: #trips up with the symbol '00-USD', 
            symbol = row['symbol'] + '-USD'
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
                    analysis = {"timedelta":str(row_count), "symbol":symbol, "commence":start_date}
                    print(f"analysis of prediction: {analysis}")
                    waits.append(analysis)
                # else:
                #     row_count = target_row_index - new_df.index[0]  # Or set to None or another appropriate value
                #     analysis_diff = {"timedelta":{str(row_count)}, "symbol": {symbol}, "commence":{start_date}}
                    # diffs.append(analysis_diff) #how many 
            else:
                print(f'no downloadable price data for symbol {symbol}')

        if row['symbol'] == 0: #trips up with the symbol '00-USD'
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
                    analysis = {"timedelta":str(row_count), "symbol": symbol, "commence":start_date}
                    print(f"waits: {analysis}")
                    waits.append(analysis)
                # else:
                #     row_count = target_row_index - new_df.index[0]  # Or set to None or another appropriate value
                #     analysis_diff = {"timedelta":{str(row_count)}, "symbol": {symbol}, "commence":{start_date}}
                #     print(f"diff: {analysis_diff}")
                    # diffs.append(analysis_diff) #how many 
            else:
                print(f'no downloadable price data for symbol {symbol}')
        #TODO: change waits_meta to waits_df 
        waits_df = pd.DataFrame(waits) #transform the list of trade durations to a dataframe
        # print(f"waits columns, {waits_df.columns}")
        # print(f"waits df content:\n {waits_df.head(5)}")
        try: #for now, reduce the 
            waits_df['timedelta'] = pd.to_timedelta(waits_df['timedelta'] ) #.str.strip('{}'))

            # Now you can safely use .total_seconds() to convert the timedelta to hours
            waits_df['hours'] = waits_df['timedelta'].dt.total_seconds() / 3600

            # waits_df.rename(columns={'timedelta': 'hours'}, inplace=True)

        #make commence to epoch, to help merge with BBPa
            # dates = list(waits_df.at[0, 'commence'])  # Convert the set to a list
            # waits_df['commence'] = [pd.to_datetime(date) for date in dates]  # Convert each date string to datetime
            # Convert the 'commence' column to datetime objects
            waits_df['commence'] = pd.to_datetime(waits_df['commence'])

            # print(f"waits meta information 1: \n{waits_df.head(1)}") #ok

            waits_df['commence'] = pd.to_datetime(waits_df['commence']) # Convert the 'commence' column to datetime
            # waits_df['commence'] = waits_df['commence'].apply(clean_and_convert) #convert commence into epoch
            # print(f"waits meta information 2: \n{waits_df.head(1)}") #ok

            waits_df['time'] = (waits_df['commence'] - pd.Timestamp('1970-01-01')).dt.total_seconds()  # Calc epoch sec
            waits_df['efficiency'] = waits_df['hours'].apply(lambda x: 1 if x < 3.0 else 0)  #hours becomes a label like qualifier
            
            combined_df = pd.concat([waits_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
            print(f"waits meta information 3: \n{combined_df.head(1)}") #ok

            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # /home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/WM-BUILD
            waits_df.to_csv(f"/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/WM-BUILD/{symbol}-WAITS-META-{current_datetime}.csv")  #waits meta data is now complete for that trade
            print("                                ")
        except Exception as e:
            print(f"waits meta dataframe issue: {e}")
    return 

def run_analysis(experiment_tmp):  # Iterate through all the files in the 'model' folder
    print(f"run analysis starts. temp directory: {experiment_tmp}")
    for filename in os.listdir(experiment_tmp):
        print(f"1, get PRED,  filename, {filename}")
        symbol = ''
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
        process_file(file_path, symbol, experiment_tmp) #supposed to dump one prediction's worth, not all
    make_duration_dataset()

    return
    
# run_analysis(init()) #determines a duration, per predicted trade in the PRED folder
make_duration_dataset()  #as of june 22, predict stores precursor features, and we get duratoin, then save 2 csv




    # except Exception as e:
    #     print(f"error running analysis {e}")
    
    

   
    #should predict the duration of a 1 class prediction, based on waits_meta? 


    # above_three_hr_df =waits_df[waits_df['hours'] > 3.0] #everything that is not desireable
    # three_hour_limit = waits_df[waits_df['hours'] < 3.0]  #three hour threshold filter, 'should be three hour filter'
    # print(f"filtereddf is of shape {three_hour_limit.shape[0]}")
#     timedelta	    symbol	        commence	        hours        duration as numeric
# 2	0 days 01:00:00	{'ONDO-USD'}	2024-05-24 23:10:44	1.000000
   
    # three_hr_merged_df = pd.merge(three_hour_limit, symboldf, left_on='epoch', right_on='time', how='right') # Merge  based on 'epoch'
    # accuracy_threshold = .88
    # symbol_meta_here ="get the string to your model, with symbol + constant" #obtain the model file, with the MIF metadata
    # model_folder = f"/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/MODEL/{symbol_meta_here}"
    # solution_df = pd.read_csv(symbol_meta_here)
    # highest_accuracy_row = solution_df[solution_df['accuracy'] == solution_df['accuracy'].max()] #most accurate model
    # #MIF, most important feature in the prediction outcome
    # most_important_feature, mif_mean, mif_std
    # mif = highest_accuracy_row['most_important_feature'].values 


#compare MIF within the 3 hour limit vs outside, as a mean
# three_hour_limit_mif = three_hour_limit[mif].mean()
# above_three_hour_limit_mif = above_three_hr_df[mif].mean()

#execute another KNN, in a separate new model area:
#DURATION_PRED folder, which contains the data needed to predict the likely duration of a total_trade
# combine keepable, y_pred, then use hours as a label, which you must bin 
# > 3 = 0
# < 3 =1 create new columns based on this evaluation

    #PRIOR CODE FROM PREDICT, WHERE PREDICTOR HANDLED THE MIF PIECES
    # most_important_feature = key_structure[max_value_index]  #MIF applied
    # mif_mean = merged_df.loc[merged_df['label'] == 1, most_important_feature].mean()
    # print(f"The mean {most_important_feature} for MIF is: {mif_mean:.2f}, as type {type(mif_mean)}")
    # print(f"final df mean: {final_df[[most_important_feature]].mean()}")
# if final_df[[most_important_feature]].mean() <= mif_mean:
    # self.
# now = datetime.now()

# insert_prediction(now.strftime("%Y-%m-%d %H:%M:%S"), y_pred.item(),self.model_meta, most_important_feature, mif_mean)
    #more recent, 175 entries
#     print(f"prediction filed") 
#     self.precursors.clear()
#     sequence_df.drop(sequence_df.index, inplace=True)
# except Exception as e:
#     print(f"FILTERED DF FAIL, couldn't create, with error, {e}")


    #after outcome_1 and outcome_0 were defined, used to use this difference machine: 
    # key_structure = []
    # for key, value in outcome_1.mean().items(): 
    #     key_structure.append(key)
    #     for key0, value0 in outcome_0.mean().items():
    #         if key == key0:
    #             print(f"{key}/optimal -> {key0}/bad -> {value - value0}")
    #             diff = value - value0
    #             max_table.append(diff) #record the distance between 1 and 0 factor values
    # max_value_index = max_table.index(max(max_table))
    # print(f" most differentiated feature for symbol is .... {key_structure[max_value_index]}") #define MIF


#transfer predictions to the trader, once clceare
# def insert_prediction(self, time, y_pred, exp, most_important_feature, mif_mean ): 
#         data = [self.symbol, time, exp, y_pred,most_important_feature, mif_mean]  #["april,2,5,7", "may,3,5,8", "june,4,7,3", "july,5,6,9"]
#         loc = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/PRED/'
#         output_csv_path = loc +self.symbol +"-prediction-log-" + time + ".csv"
#         with open(output_csv_path, "w", newline="") as csv_file:
#             self.writer = csv.writer(csv_file)  # Create a CSV writer object
#             self.writer.writerow(data)  # Split the line by comma and write it to the CSV
#         print(f"prediction filed") 
#         #run scp on scp_target_folder
#         scp_path =  loc + "'" +self.symbol +"-prediction-log-" + time + ".csv" + "'"
#         scp_target_folder = '/home/stefan/Desktop/jancula_tests/stadiumBoulevardTrader/working-folder-may-2024/PRED'
#         scp_command = "scp " + scp_path +" stefan@192.168.6.118:" +scp_target_folder
#         subprocess.run(scp_command, shell=True)
#         return