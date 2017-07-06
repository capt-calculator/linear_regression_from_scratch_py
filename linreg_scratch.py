import numpy as np
import pandas as pd
from numpy.linalg import inv
from sklearn.datasets import load_boston
from statsmodels.regression.linear_model import OLS

# load the boston data set
boston = load_boston()

# obtain the feature matrix as a numpy array
X = boston.data

# obtain the target variable as a numpy array
y = boston.target

# create vector of ones...
int = np.ones(shape=y.shape)[..., None]

#...and add to feature matrix
X = np.concatenate((int, X), 1)

# calculate coefficients using closed-form solution
coeffs = inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

# extract the feature names of the boston data set and prepend the intercept
feature_names = np.insert(boston.feature_names, 0, 'INT')

# collect results into a DataFrame for pretty printing
results = pd.DataFrame({'coeffs':coeffs}, index=feature_names)

# create a linear model and extract the parameters
coeffs_lm = OLS(y, X).fit().params

# add the coefficients to the results DataFrame
results['coeffs_lm'] = coeffs_lm

print(results.round(2))






print(results.round(2))
