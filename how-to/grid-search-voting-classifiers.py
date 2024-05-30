import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset (you can replace this with your own data)
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a list of classifiers
clf1 = LogisticRegression()
clf2 = SVC(kernel='linear', probability=True)
clf3 = DecisionTreeClassifier()

# Create a dictionary of hyperparameters for each classifier
param_grid = {
    'clf1__C': [0.1, 1, 10],
    'clf2__C': [0.1, 1, 10],
    'clf3__max_depth': [3, 5, 7]
}

# Create the voting classifier
voting_clf = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('dt', clf3)], voting='soft')

# Perform grid search using GridSearchCV
grid_search = GridSearchCV(estimator=voting_clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

# Evaluate the model on the test set
best_voting_clf = grid_search.best_estimator_
test_accuracy = best_voting_clf.score(X_test, y_test)
print("Test accuracy:", test_accuracy)
