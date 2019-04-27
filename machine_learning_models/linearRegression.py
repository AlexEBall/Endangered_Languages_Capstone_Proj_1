# General imports
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

# model selection
from sklearn.linear_model import LinearRegression

# plotting config
sns.set()
%matplotlib inline

# import data
endangered_languages = pd.read_csv('./data_sets/endangered_languages_ML.csv')
endangered_languages.head()

# View shape, get X & Y
X = endangered_languages.drop(['Language', 'Extinct'], axis=1)
y = endangered_languages[['Extinct']]
print(X.shape)
print(y.shape)

# instantiate model
lm = LinearRegression()

# perform k-fold cross-validation (which takes a holdout set from each interval)
cv_results = cross_val_score(lm, X, y, cv=5)
# then fit an predict as usual


# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# fit your model, which does Ordinary Least Squares (OLS) nder the hood
lm.fit(X_train, y_train)

# Look at coefiecents
coeff_df = pd.DataFrame(flattened_coef, X.columns, columns=['Coefficient'])

# predict
y_pred = lm.predict(X_test)

# Analyze
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
print('Score: {}'.format(lm.score(X_test, y_test)))

# 
# ridge regression - which will select the best alpha/lambda to fit loss funciton: OLS r2 prediction + a (for ai in n) ai2n)

ridge = Ridge(normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print(ridge.score(X_test, y_test))

# try ridge with alpha croxx-validated
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error,
                    cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X_train, y_train, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)

# or Lasso Regression
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print(lasso.score(X_test, y_test))

# Lasso predicting best features
X_features = endangered_languages.drop(['Language', 'Extinct'], axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X_train, y_train).coef_
_ = plt.plot(range(len(X_features)), lasso_coef)
_ = plt.xticks(range(len(X_features)), X_features, rotation=60)
_ = plt.ylabel('Coeficients')
plt.show()

# plot scatter 
## The line / model
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')

# plot residulas
# Residuals plot
_ = plt.scatter(lm.predict(X_train), lm.predict(
    X_train) - y_train, c='b', s=40, alpha=0.5)
_ = plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test, c='g', s=40)
_ = plt.hlines(y=0, xmin=-5, xmax=5)
_ = plt.title('Residual Plot using training (blue) and test(green) data')
_ = plt.ylabel('Residuals')
plt.show()
