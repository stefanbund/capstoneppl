#COPYRIGHT STEFAN L. BUND, copyright available at https://github.com/stefanbund/radidisco. No use is authorized and no sharing license is granted.
# reflects the need to insert a prediction with a full precursor, used to predict. So the first outcome is predicted versus BBP, then a 
# duration prediction made against the WAITS-META figures, for that symbol
import pandas as pd
import numpy as np
import os
import sched, time, threading
import csv
from datetime import date, datetime
from imblearn.over_sampling import ADASYN
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
import csv
import subprocess
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA


def build_classifier(classifier_name, params): 
    if classifier_name == 'LogisticRegression':
        classifier = LogisticRegression(**params)
    if classifier_name == 'KNeighborsClassifier':
        classifier = KNeighborsClassifier(**params)
    if classifier_name == 'BernoulliNB':
        classifier = BernoulliNB(**params)
    return  classifier

def build_BERT():
    #load torch model, per Xioaolan

    return 0

'''
loads the model file from our folder for models, one at a time
loads the BBP.csv data, for that model'''
def load_model(filename, modelDictionary):  
    symbol  = filename[:filename.index("-")]  #locate symbol for BBP
    bbp_location = "/home/stefan/Desktop/STADIUM-DATA/BBP/" + symbol + "-binary_binned_pipeline.csv"  #get from BBP folder
    # print(f"load model with filename {filename}, for symbol, {symbol}, at {bbp_location}")
    m2_pipeline = pd.read_csv(bbp_location)  #get BBP for symbol
    m2_pipeline['time'] = m2_pipeline['time'].ffill() + 100 #.fillna(method='ffill') + 100
    keepable = ['precursor_buy_cap_pct_change', 'precursor_ask_cap_pct_change',
                    'precursor_bid_vol_pct_change',  'precursor_ask_vol_pct_change',
                    'sum_change','length','time'] 
    y = m2_pipeline['label'].values 
    X = m2_pipeline[keepable].values
    X_resampled, y_resampled = ADASYN(random_state=42 ).fit_resample(X, y)  
    scaler = StandardScaler()   
    X_train_scaled = scaler.fit_transform(X_resampled)
    best_params =modelDictionary['best_params'].to_dict()
    params_str = best_params.get(0, "") # Extract the string value associated with key 0
    try:    # Convert the string to a dictionary
        extracted_dict = eval(params_str)
    except SyntaxError:
        extracted_dict = {}
    classifier_name= str( modelDictionary['classifier'].iloc[0])  
    classifier_x = build_classifier(classifier_name, extracted_dict)
    # print(symbol, classifier_x) 
    if classifier_name == 'LogisticRegression':  #new as of June 29th, using NCA
        # Initialize the NeighborhoodComponentsAnalysis 
        nca = NeighborhoodComponentsAnalysis(n_components=7) # You can choose the number of components 
        # Fit and transform the data using NCA 
        X_train_nca = nca.fit_transform(X_train_scaled, y_resampled) 
        X_test_nca = nca.transform(X_resampled) 
        return classifier_x.fit(X_train_nca, y_resampled)
    else: 
        return classifier_x.fit(X_train_scaled, y_resampled) #for those not using NCA, only Scaler, ADASYN
class Predictor:
    def __init__(self, symbol, model_meta,meanChange ):
        self.model_meta = model_meta
        self.symbol = symbol #from caller
        self.in_in = 0
        self.threshold =meanChange #meanChange value
        self.temp_caps_df = pd.DataFrame() #will be used in several functions
        self.clf_loaded = model_meta #descriptive row, highest accuracy model
        self.precursors = []
        self.last_id = 0
        self.my_scheduler = sched.scheduler(time.time, time.sleep) #make pred
        self.predictions_df = pd.DataFrame()
        self.thread = threading.Thread(target=self.run_scheduler) #make Prediction
        self.thread.start()
        self.base_loc = '/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/'
        self.desktop_offload = '/home/stefan/Desktop/STADIUM-DATA/ERROR_LOGS/'
        self.directional_pred_folder = '/home/stefan/Desktop/STADIUM-DATA/DIRECTIONAL_PREDICTIONS/'
        self.send_to_trader = '/home/stefan/Desktop/STADIUM-DATA/SCP-PRED/'
                                          #/home/stefan/Desktop/STADIUM-DATA/SYMBOL_DURATIONS_FITTABLE/POWR-USD-waits-meta-data.csv
        self.duration_prediction_folder = '/home/stefan/Desktop/STADIUM-DATA/SYMBOL_DURATIONS_FITTABLE/'


    def run_scheduler(self):  #create the csv file and the scheduled prediction, ie make prediction, below
        self.makePrediction()
        self.my_scheduler.run()                 # print(f"{self.symbol} initiated")

    def insert_prediction(self, time, y_pred, exp, buy_cap, ask_cap, bid_vol, ask_vol, sum_change, length): 
        data = [self.symbol, time, exp, y_pred, buy_cap, ask_cap, bid_vol, ask_vol, sum_change, length]  #6-22, we will take latest precursor, and couple with duration in WM
        # loc = '/home/stefan/Desktop/STADIUM-DATA/DIRECTIONAL_PREDICTIONS/'#'/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/PRED/'
        output_csv_path = self.directional_pred_folder +self.symbol +"-prediction-log-" + time + ".csv"
        with open(output_csv_path, "w", newline="") as csv_file:
            self.writer = csv.writer(csv_file)  # Create a CSV writer object
            self.writer.writerow(data)  # Split the line by comma and write it to the CSV
        print(f"{self.symbol} prediction filed") 
        return 
    
    def send_trade_for_prediction(self, time, y_pred, model_meta, precursor_buy_cap_pct_change, precursor_ask_cap_pct_change, precursor_bid_vol_pct_change,  
                precursor_ask_vol_pct_change,sum_change,length, duration_y_pred ):  #save to csv and send via scp as we normally would, to traders
        try: 
            data = [time, y_pred, model_meta, precursor_buy_cap_pct_change, precursor_ask_cap_pct_change,precursor_bid_vol_pct_change,  
                    precursor_ask_vol_pct_change,sum_change,length, duration_y_pred] 
            # loc = '/home/stefan/Desktop/STADIUM-DATA/SCP-PRED/'
            output_csv_path = self.send_to_trader +self.symbol +"-prediction-log-" + time + ".csv"
            with open(output_csv_path, "w", newline="") as csv_file:
                self.writer = csv.writer(csv_file)  # Create a CSV writer object
                self.writer.writerow(data)  # Split the line by comma and write it to the CSV
            print(f"{self.symbol} DURATION enabled prediction filed") 
        except Exception as e:
            print(f"{self.symbol} send final trade rec with DURATION error: {e}")
        #run scp on scp_target_folder
        scp_path =  self.send_to_trader + "'" +self.symbol +"-prediction-log-" + time + ".csv" + "'"
        scp_target_folder = '/home/stefan/Desktop/jancula_tests/stadiumBoulevardTrader/working-folder-may-2024/PRED'
        scp_command = "scp " + scp_path +" stefan@192.168.6.118:" +scp_target_folder
        # Run the SCP command
        subprocess.run(scp_command, shell=True)
        return 
                   
    def predictForCompiledPrecursor(self):              #predict, only once the market turns from precursor to surge
        sequence_df = pd.DataFrame(self.precursors)      #prepare to process the precursors
        keepable = ['precursor_buy_cap_pct_change', 
                'precursor_ask_cap_pct_change',
                'precursor_bid_vol_pct_change', 
                'precursor_ask_vol_pct_change',
                'sum_change','length','time']
        final_df = pd.DataFrame(columns=keepable)
        final_df.at[0, 'precursor_buy_cap_pct_change'] = sequence_df['precursor_buy_cap_pct_change'].sum()
        final_df.at[0, 'precursor_ask_cap_pct_change'] = sequence_df['precursor_ask_cap_pct_change'].sum()
        final_df.at[0, 'precursor_bid_vol_pct_change'] = sequence_df['precursor_bid_vol_pct_change'].sum()
        final_df.at[0, 'precursor_ask_vol_pct_change'] = sequence_df['precursor_ask_vol_pct_change'].sum()
        final_df.at[0, 'sum_change'] = sequence_df['change'].sum()
        final_df.at[0, 'length'] = sequence_df.shape[0]
        final_df.at[0, 'time'] = sequence_df['change'].max()
        # print(f"{self.symbol} ")#  final df:", final_df)
        X = final_df.values  
        y_pred = self.clf_loaded.predict(X)
        # print(type(y_pred))
        now = datetime.now()
        # print(now.strftime("%Y-%m-%d %H:%M:%S"), "PREDICTED CLASS:",y_pred)
        if(y_pred.item() == 1):
            self.insert_prediction(now.strftime("%Y-%m-%d %H:%M:%S"), 
                                   y_pred.item(),self.model_meta,final_df.at[0, 'precursor_buy_cap_pct_change'],final_df.at[0, 'precursor_ask_cap_pct_change'],
                                   final_df.at[0, 'precursor_bid_vol_pct_change'] ,final_df.at[0, 'precursor_ask_vol_pct_change'],final_df.at[0, 'sum_change'],
                                   final_df.at[0, 'length'])#insert into PRED, feature-rich, where we analye patternific execution, and duration
            self.precursors.clear()
            sequence_df.drop(sequence_df.index, inplace=True)
            final_df.at[0, 'y_pred'] = 1 #because we predicted 1
            try: 
                # na_values = ['NaN', 'N/A', ' ']
                print(f"final df columns names: {final_df.columns}")
                #     final df columns names: Index(['precursor_buy_cap_pct_change', 'precursor_ask_cap_pct_change',
                #    'precursor_bid_vol_pct_change', 'precursor_ask_vol_pct_change',
                #    'sum_change', 'length', 'time', 'y_pred']
                fittable_df = pd.read_csv(f"{self.duration_prediction_folder}{self.symbol}-USD-waits-meta-data.csv") #was TESTLIST
                wm = ['precursor_buy_cap_pct_change', 'precursor_ask_cap_pct_change','precursor_bid_vol_pct_change',  
                    'precursor_ask_vol_pct_change','sum_change','length']  #,'bin'] TESTLIST format, y_pred is a constant of 1
                print(f"START NCA: TESTLIST FILE: \n {self.duration_prediction_folder}{self.symbol}-USD-waits-meta-data.csv")
                # fittable_df =fittable_df.drop(columns=['buy_cap', 'ask_cap', 'bid_vol', 'ask_vol']) 
                # List of columns to check
                columns_to_drop = ['buy_cap', 'ask_cap', 'bid_vol', 'ask_vol']

                # Check if the columns exist in the dataframe and drop them
                fittable_df = fittable_df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

                fittable_df = fittable_df.dropna()
                usable_df = fittable_df[wm] #.drop(columns=['buy_cap', 'ask_cap', 'bid_vol', 'ask_vol']) #nan values
                print(f"\tfittable NCA: \n{usable_df}")
                nca = NeighborhoodComponentsAnalysis(n_components=6)
                nca.fit(usable_df.values, fittable_df['bin'])  #ok, from example
                X_transformed = nca.transform(usable_df.values)  #on keepable columns
                knn = KNeighborsClassifier(algorithm='auto', n_neighbors=11, weights='distance')     #(n_neighbors=3)
                #Best Parameters: {'algorithm': 'auto', 'n_neighbors': 11, 'weights': 'distance'}

                knn.fit(X_transformed, fittable_df['bin'])  #trying

                duration_y_pred = knn.predict(final_df[wm].values)              #needs same column names as fittable
                print(f"\tDURATION PREDICTION: {duration_y_pred}")
                if duration_y_pred.item() == 1:
                    self.send_trade_for_prediction(now.strftime("%Y-%m-%d %H:%M:%S"), y_pred.item(),self.model_meta,
                                    final_df.at[0, 'precursor_buy_cap_pct_change'], final_df.at[0, 'precursor_ask_cap_pct_change'],
                                    final_df.at[0, 'precursor_bid_vol_pct_change'] , final_df.at[0, 'precursor_ask_vol_pct_change'],
                                    final_df.at[0, 'sum_change'],final_df.at[0, 'length'], duration_y_pred.item()) #o scp of the trade as we normally would
            except Exception as e:
                print(f"\tnca creation and send to csv error, {e}")
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                dict = {"time":timestamp,"symbol":self.symbol, "area":"predictForCompiledPrecursor","error":e}
                err_df = pd.DataFrame(dict,index=[0])
                print(f"\tsaving error log to csv, {self.desktop_offload}{self.symbol}-{timestamp}-error_log.csv")
                err_df.to_csv(f"{self.desktop_offload}{self.symbol}-{timestamp}-error_log.csv")
        return

    def initial_intake(self):  #GET NEWEST FILE, OR TOP TWO NEWEST FILES? INITIAL INTAKE
        # dir_path = '/home/stefan/Desktop/GRUS-CSV-SAMPLER-DATA' #global cell-2024, or lob_Caps equiv in this igteratgion
        # '/Users/stefanbund/Desktop/Desktop - stefan’s Mac mini/marine/marine/1-6-post-hft-AVAX/lob_caps'
        form = self.symbol + "-USD"  #AVAX-USD
        # print(form)
        ret = [] #FORMERLY WALKING THE LOB_CAPS FOLDER, now loads all cell2024-x.csv files
        # Step 1: List files starting with "cell"
        directory = "/home/stefan/Desktop/GRUS-CSV-SAMPLER-DATA/"  #os.getcwd()  # Get current directory
        def get_name_of_newest_csv():
            file_list = [filename for filename in os.listdir(directory) if filename.startswith("cell")]
            newest_file = max(file_list, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
            return directory + newest_file
        newest_file = get_name_of_newest_csv() #just get the latest data on the symbol, from the cell
        #assume the totality of cell data has been assembled into the MODEL solution...
        provisional = pd.read_csv(newest_file,index_col=None, header=0)  # Assuming CSV files, adjust as needed
        symbol_rows = provisional[provisional["symbol"] == form] # Step 3: Search for rows where "symbol" matches the value of "pair"
        self.in_in = 1 #flag set to 1 once we have entered the backlog of precursors, start reading tail(1)
        return symbol_rows  #pd.read_csv(name) #return a dataframe containing latest data

    def get_tail_latest(self):
        dir_path = '/home/stefan/Desktop/GRUS-CSV-SAMPLER-DATA'
        #'/Users/stefanbund/Desktop/Desktop - stefan’s Mac mini/marine/marine/1-6-post-hft-AVAX/lob_caps'#now cell-2024
        def get_name_of_newest_csv():   
            caps_files = [f for f in os.listdir(dir_path) if f.endswith('CAPS.csv')]
            newest_file = max(caps_files, key=lambda f: os.path.getmtime(os.path.join(dir_path, f)))
            return newest_file 
        name = get_name_of_newest_csv()
        name = dir_path +"/"+get_name_of_newest_csv()
        latest = pd.read_csv(name).tail(20)
        # print("GET TAIL LATEST:",latest.columns)
        return latest                             #just the last, theoretically latest entry

    def make_initial_df(self): #take the data from csv, classify it then populate the most recent precursorslobal 
        lookback_period = 10  
        self.temp_caps_df = self.initial_intake() 
        last_id = self.temp_caps_df.index[-1]
        self.temp_caps_df['change'] = self.temp_caps_df['mp'].pct_change(periods=lookback_period) 
        self.temp_caps_df['bc_change'] = self.temp_caps_df['bc'].pct_change(periods=lookback_period)
        self.temp_caps_df['ac_change'] = self.temp_caps_df['ac'].pct_change(periods=lookback_period)
        self.temp_caps_df['tav_change'] = self.temp_caps_df['tav'].pct_change(periods=lookback_period)
        self.temp_caps_df['tbv_change'] = self.temp_caps_df['tbv'].pct_change(periods=lookback_period)
        for i in range(0,len(self.temp_caps_df),10):                                         #once we reach 10 rows
            # print("change:", self.temp_caps_df.iloc[i:i+10]['change'].mean()) #11/4 mean of rows in sampling
            if self.temp_caps_df.iloc[i:i+10]['change'].mean() <= self.threshold:
                self.precursors.append({'time': self.temp_caps_df.iloc[i]['time'], 'p_MP': self.temp_caps_df.iloc[i]['mp'],
                                'change': self.temp_caps_df.iloc[i:i+10]['change'].mean(), 'type':'precursor',
                                'precursor_buy_cap_pct_change':self.temp_caps_df.iloc[i]['bc_change'], 
                                'precursor_ask_cap_pct_change':self.temp_caps_df.iloc[i]['ac_change'],
                                'precursor_bid_vol_pct_change':self.temp_caps_df.iloc[i]['tbv_change'],
                                'precursor_ask_vol_pct_change':self.temp_caps_df.iloc[i]['tav_change']}) 
            else:
                # print("make_initial_df: surge detect, clearing precursors list, DO NOT PREDICT") 
                self.precursors.clear()
                # print("LENGTH OF PRECURSOR LIST IS ", len(globals()['precursors']))
        # print("last id starts at", last_id,"row:", self.temp_caps_df.iloc[-1])
        self.temp_caps_df.drop(self.temp_caps_df.index, inplace=True)
        # return precursors #?

    def get_last_index(self):
        dir_path = '/home/stefan/Desktop/GRUS-CSV-SAMPLER-DATA'
        def get_name_of_newest_csv():   
            caps_files = [f for f in os.listdir(dir_path) if f.endswith('CAPS.csv')]
            newest_file = max(caps_files, key=lambda f: os.path.getmtime(os.path.join(dir_path, f)))
            return newest_file 
        name = get_name_of_newest_csv()
        name = dir_path +"/"+get_name_of_newest_csv()
        latest = pd.read_csv(name)
        last_value = latest.index[-1]  #.astype(int)
        # print("GET TAIL LATEST:",last_value) #ok
        return last_value

    def justify_classify_event(self):   #look at the most recent observation, and check its index, if index = last +10, return true
        # global last_id
        last = self.get_last_index()
        # print("last id was ", last_id, "vs latest row,", last)#ok
        result = False
        if last >= self.last_id + 10:  #will miss if the last row is suddenly more
            result =  True
            # print("just classify is ", result)
            self.last_id = last
            # print("last id set to ",self.last_id)
        return result

    def makePrediction(self):
        # print(f'make prediction with {self.symbol}')  #working from scheduler call in init
        self.my_scheduler.enter(10, 1, self.makePrediction)

        # self.my_scheduler.enter
        if self.in_in ==0:      
            self.temp_caps_df = self.make_initial_df()
        if self.justify_classify_event():   #if true, go forward else dump
            # print("make prediction: just classify is true")
            # caps_df = pd.concat([caps_df, get_tail_latest(20)], ignore_index=True)  #refers to latest entry, latest file
            self.temp_caps_df = self.get_tail_latest()          #reflect 10 observations back, in order to do operation below, lookback
            # self.temp_caps_df.ffill()
            lookback_period = 10  
            # threshold = meanChange  # in rows
            self.temp_caps_df.ffill()# fillna(method='ffill', inplace=True) #Use obj.ffill() or obj.bfill() instead.
            # print("10 caps processing...",len(self.temp_caps_df))
            self.temp_caps_df['change'] = self.temp_caps_df['mp'].pct_change(periods=lookback_period) #setup 
            self.temp_caps_df['bc_change'] = self.temp_caps_df['bc'].pct_change(periods=lookback_period)
            self.temp_caps_df['ac_change'] = self.temp_caps_df['ac'].pct_change(periods=lookback_period)
            self.temp_caps_df['tav_change'] = self.temp_caps_df['tav'].pct_change(periods=lookback_period)
            self.temp_caps_df['tbv_change'] = self.temp_caps_df['tbv'].pct_change(periods=lookback_period) #always unitary value for each, always an agg
            self.temp_caps_df = self.temp_caps_df.tail(10)                #necessary, else 10 nans, which spoils the knn predict
            self.temp_caps_df.ffill()#   fillna(method='ffill', inplace=True)
            # print( temp_caps_df)
            # print("change:", self.temp_caps_df['change'].mean())         #duplicate data processing steps to best ability
            for i in range(0, len(self.temp_caps_df),10):                 #replicate data prep step
                if self.temp_caps_df['change'].mean() <= self.threshold:      #test whether price change resembles a surge
                    self.precursors.append({'time': self.temp_caps_df.iloc[i]['time'],'p_MP': self.temp_caps_df.iloc[i]['mp'], #add a precursor event
                                    'change': self.temp_caps_df['change'].mean(), 'type':'precursor',
                                    'precursor_buy_cap_pct_change':self.temp_caps_df.iloc[i]['bc_change'], 
                                    'precursor_ask_cap_pct_change':self.temp_caps_df.iloc[i]['ac_change'],
                                    'precursor_bid_vol_pct_change':self.temp_caps_df.iloc[i]['tbv_change'],
                                    'precursor_ask_vol_pct_change':self.temp_caps_df.iloc[i]['tav_change']}) 
                    # print("ADDED A PRECURSOR, NEW LENGTH:",len(self.precursors))
                    return
                else:
                    # print("SURGE DETECTED...PREDICT FOLLOWS") 
                    # print("LENGTH OF PRECURSOR LIST IS ", len(self.precursors))
                    self.temp_caps_df.drop(self.temp_caps_df.index, inplace=True)                   # clear out the caps dataframe
                    # print("CAPS_DF NOW ", self.temp_caps_df.shape[0])
                    # print( temp_caps_df)
                    if len(self.precursors) != 0:
                        self.predictForCompiledPrecursor() 
                    # else:
                    #     print("surge but with empty precursors...")
        return  
                       
try:
    while True:  # This creates an infinite loop
        accuracy_threshold = .88
        # current_directory = os.path.dirname(os.path.abspath(__file__))
        model_folder = '/home/stefan/Desktop/STADIUM-DATA/MODEL-STADIUM' #"/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/MODEL"
        #"/home/stefan/Desktop/MARCH-2024-RADDISCO-GH-REPO/radDisco-recon/cell-2024/MODEL"
        #"./MODEL"  #os.path.join(current_directory, "MODEL")

        predictors = []
        # Iterate over CSV files in the "MODEL" folder
        for filename in os.listdir(model_folder):
            if filename.lower().endswith(".csv"):
                meanChange = 0.00055449     #must set after each manual model take 
                symbol = filename.split('-')[0]  # Extract the first characters before the symbol '-'
                # print(f"The extracted part before the symbol '-' is: {symbol}")  # Print the extracted part, SYMBOL defines the loop
                csv_path = os.path.join(model_folder, filename)   #exract the symbol name here, for use in initialIntake, in makePrediction
                # print(f"MODEL FILE: {csv_path}")
                df = pd.read_csv(csv_path)
                highest_accuracy_row = df[df['accuracy'] == df['accuracy'].max()]
                if highest_accuracy_row['accuracy'].values >= accuracy_threshold:
                    # precursors = []
                    clf_loaded = load_model(filename, highest_accuracy_row) #will extract the symbol from the filename                    
                    p = Predictor(symbol, clf_loaded, meanChange)  #send symbol and model meta data
                    predictors.append(p)
                time.sleep(1)
except KeyboardInterrupt:
    print("Stopping all schedulers...")
    for predictor in predictors:
        predictor.scheduler.cancel()
    for predictor in predictors:
        predictor.thread.join()
    print("All schedulers stopped.")