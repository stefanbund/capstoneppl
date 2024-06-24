
from datetime import date
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE,SMOTENC, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import pandas as pd 
import numpy as np

from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import BalancedBaggingClassifier,BalancedRandomForestClassifier,RUSBoostClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier, \
ExtraTreesClassifier, RandomTreesEmbedding, BaggingClassifier
from sklearn.linear_model import RidgeCV, LassoCV

import os
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from scipy import stats

from sklearn.tree import DecisionTreeClassifier
import shap
import time
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  VotingClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
from sklearn.inspection import DecisionBoundaryDisplay
import pickle

folder_path = './BBP'  #where we source sample data, for each symbol

# The directory where your current files are located
source_directory = "/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/MODEL"


# The directory where you want to move the files
destination_directory = "/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/MODEL/HISTORICAL"

# Get today's date in the format YYYYMMDD
today_date = datetime.now().strftime('%Y%m%d')

# Create the HISTORICAL_MODELS directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Rename and move each file in the source directory
for filename in os.listdir(source_directory):
    # Skip directories
    if os.path.isdir(os.path.join(source_directory, filename)):
        continue

    # Create the new filename by adding today's date
    new_filename = f"{today_date}_{filename}"

    # The full path for the source and destination
    source_path = os.path.join(source_directory, filename)
    destination_path = os.path.join(destination_directory, new_filename)

    # Move and rename the file
    shutil.move(source_path, destination_path)

print("Files have been renamed and moved successfully.")

for filename in os.listdir(folder_path):  #INITIATE MODEL BUILD, PER SYMBOL
    if filename.endswith('.csv'):
        original_string = filename
        index_of_dash = original_string.find('-')
        if index_of_dash != -1:
            substring_before_dash = original_string[:index_of_dash]
        else:
            substring_before_dash = ""
        file_path = os.path.join(folder_path, filename)
        m2_pipeline = pd.read_csv(file_path)
        symbol = substring_before_dash
        

        if len(m2_pipeline['label'].unique()) >= 2: #if we can't see more than one class, don't model
            m2_pipeline.drop(m2_pipeline.tail(1).index, inplace=True)

            keepable = ['precursor_buy_cap_pct_change','precursor_ask_cap_pct_change',
                    'precursor_bid_vol_pct_change', 'precursor_ask_vol_pct_change','sum_change','length','time']
            X = m2_pipeline[keepable].values
            y = m2_pipeline['label'].values
            try:
                X_resampled, y_resampled = ADASYN(random_state=42 ).fit_resample(X, y)
                X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
                scaler = StandardScaler() 
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.fit_transform(X_test)
                classifiers = [  
                    LogisticRegression(),
                    BernoulliNB(),
                    KNeighborsClassifier(),
                ]    
                params = {'LogisticRegression':{'C': [0.1, 1, 10], 'penalty':['l1','l2','elasticnet','None'], 'multi_class':['ovr','auto'],\
                                            'random_state':[42]},
                    'BernoulliNB':{'fit_prior':[True, False]},
                    'KNeighborsClassifier':{'n_neighbors':[3,4,5,6,7,8], 'algorithm':['auto'], 'n_jobs':[1,2,3,4]} }
                comparative = []
                for clf in classifiers:
                    name = clf.__class__.__name__
                    if name in params:
                        grid_search = GridSearchCV(clf, params[name], cv=5)
                        grid_search.fit(X_train_scaled, y_train)
                        accuracy = grid_search.score(X_test_scaled, y_test)
                        dict = {"classifier":name, "best_params":grid_search.best_params_, "accuracy":accuracy}
                        comparative.append(dict)
                solution_df = pd.DataFrame(comparative)
                d4 = date.today().strftime("%B %d, %Y")                
                accuracy_threshold = .88
                model_folder = "/home/stefan/Desktop/raddisco-github-repo/radDisco-recon/cell-2024/MODEL"
                highest_accuracy_row = solution_df[solution_df['accuracy'] == solution_df['accuracy'].max()]
                if highest_accuracy_row['accuracy'].values >= accuracy_threshold:

                    #fit best performing model here, get most key features, and store in model file
                    solution_df.to_csv(f'{model_folder}/{symbol}-model.csv', index=False)
                else:
                    print(f"Accuracy for {symbol} does not meet the threshold.")

            except Exception as e:
                print(f"{symbol} --An error occurred: {e}")