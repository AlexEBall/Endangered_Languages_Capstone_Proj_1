# -------------------------------------
#       Imports
# -------------------------------------

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score

sns.set()
%matplotlib inline

endangered_languages = pd.read_csv('./data_sets/endangered_languages_ML.csv')
endangered_languages.head()

endangered_languages.describe()

# -------------------------------------
#   Set up X (features) and y (labels)
# -------------------------------------

# Labels are the values we want to predict (don't load it as a df, just a series)
y = np.array(endangered_languages['Extinct'])

# Remove the labels from the features
# axis 1 refers to the columns
X = endangered_languages.drop(['Language', 'Extinct'], axis=1)

# Saving feature names for later use
X_names = list(X.columns)

# Convert to numpy array
X = np.array(X)


# -------------------------------------
#           Split and Train
# -------------------------------------

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# -------------------------------------
#     Instantiate, fit and predict
# -------------------------------------

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

# Train the model on training data
rf.fit(X_train, y_train)

# Use the forest's predict method on the test data
predictions = rf.predict(X_test)


# -------------------------------------
#               Metrics
# -------------------------------------

# Calculate the absolute errors
errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')



# -------------------------------------
#       Finding Important Values
# -------------------------------------

# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_names, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# -----------------------------------------------
#  Rerun model with just two important features
# -----------------------------------------------

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators=1000, random_state=9)


# Extract the two most important features
important_indices = [X_names.index(
    'Speakers'), X_names.index('Critically endangered')]
train_important = X_train[:, important_indices]
test_important = X_test[:, important_indices]


# Train the random forest
rf_most_important.fit(train_important, y_train)


# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)

# -------------------------------------
#            Rerun Metrics
# -------------------------------------

errors = abs(predictions - y_test)

# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / y_test))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')


# -------------------------------------
#       Graph import variables
# -------------------------------------

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation='vertical')

# Tick labels for x axis
plt.xticks(x_values, X_names, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')
