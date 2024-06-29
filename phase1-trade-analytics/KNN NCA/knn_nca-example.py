import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import NeighborhoodComponentsAnalysis 
from imblearn.over_sampling import ADASYN 

 
 

# Assuming m2_pipeline is your DataFrame 

# Replace this with your actual data 

keepable = ['precursor_buy_cap_pct_change', 
'precursor_ask_cap_pct_change', 
'precursor_bid_vol_pct_change', 
'precursor_ask_vol_pct_change', 
'sum_change', 'length', 'time'] 

 
 

y = m2_pipeline['label'].values # y vector, a list of all labeled instances 

X = m2_pipeline[keepable].values # X matrix of all features 

 
 

# Create synthetic classes using ADASYN 

X_resampled, y_resampled = ADASYN(random_state=42).fit_resample(X, y) 

 
 

# Split the data into training and testing sets 

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42) 

 
 

# Standardize the features 

scaler = StandardScaler() 

X_train_scaled = scaler.fit_transform(X_train) 

X_test_scaled = scaler.transform(X_test) 

 
 

# Initialize the NeighborhoodComponentsAnalysis 

nca = NeighborhoodComponentsAnalysis(n_components=2) # You can choose the number of components 

 
 

# Fit and transform the data using NCA 

X_train_nca = nca.fit_transform(X_train_scaled, y_train) 

X_test_nca = nca.transform(X_test_scaled) 

 
 

# Initialize the KNeighborsClassifier 

knn_classifier = KNeighborsClassifier(n_neighbors=3) # You can choose the number of neighbors 

 
 

# Fit the model to the transformed training data 

knn_classifier.fit(X_train_nca, y_train) 

 
 

# Make predictions on the transformed test data 

y_pred = knn_classifier.predict(X_test_nca) 

 
 

# Evaluate the model 

accuracy = accuracy_score(y_test, y_pred) 

conf_matrix = confusion_matrix(y_test, y_pred) 

class_report = classification_report(y_test, y_pred) 

 
 

# Print the results 

print(f"Accuracy with NCA: {accuracy:.2f}") 

print(f"Confusion Matrix:\n{conf_matrix}") 

print(f"Classification Report:\n{class_report}") 