#Classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  #Prediction error metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV, Ridge, RidgeCV, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

REGMODELS = []
REGMODELS.append(("LinearRegression",LinearRegression()))
REGMODELS.append(("RidgeRegression",Ridge()))
REGMODELS.append(("LassoRegression",Lasso()))
REGMODELS.append(("LassoRegressionAlpha1.0",Lasso(alpha=1.0)))
REGMODELS.append(("LassoRegressionAlpha0.5",Lasso(alpha=0.5)))
REGMODELS.append(("LassoRegressionAlpha0.1",Lasso(alpha=0.5)))
REGMODELS.append(("ElasticNet",ElasticNet()))
REGMODELS.append(("DecisionTree",DecisionTreeRegressor()))
#MODELS.append(("Support Vector Regression",svm.SVR())) #NOT WORKING
REGMODELS.append(("Gradient Boosting Regression",GradientBoostingRegressor()))
REGMODELS.append(("Random Forrest Regressor",RandomForestRegressor()))

CLMODELS  = []
CLMODELS.append(("k_nearest_neighbors_10", KNeighborsClassifier(10)))
CLMODELS.append(("svm_linear",SVC(kernel="linear", C=0.025, probability=True)))
CLMODELS.append(("svm_nonlinear",SVC(gamma=2, C=1, probability=True)))
CLMODELS.append(("decision_tree",DecisionTreeClassifier()))
CLMODELS.append(("naive_bayes", GaussianNB()))
CLMODELS.append(("random_forest",RandomForestClassifier()))
CLMODELS.append(("adaboost",AdaBoostClassifier()))
CLMODELS.append(("quadratic_discriminant_analysis",QuadraticDiscriminantAnalysis()))

import Helper
import Score
import Aws
import pandas as pd
import numpy as np
from datetime import datetime
from IPython.core.display import display

def perform_regression(cf, train_X, test_X, train_y, test_y, models = REGMODELS):
    train_predict=pd.DataFrame(index=train_X.index)
    test_predict=pd.DataFrame(index=test_y.index)
    #Include ground truth in prediction file.
    train_predict['target']=train_y
    test_predict['target']=test_y

    trained_models=[]

    for name, model in models:
        print("Fitting model: ", name )
        m = model.fit(train_X, train_y)

        train_predict[name]=m.predict(train_X)

        test_predict[name]=m.predict(test_X)
        trained_models.append({name: m})

    return train_predict, test_predict, trained_models


def perform_classification(train_X, test_X, train_y, test_y, models = CLMODELS):
    train_predict=pd.DataFrame(index=train_X.index)
    test_predict=pd.DataFrame(index=test_y.index)
    #Include ground truth in prediction file.
    train_predict['target']=train_y
    test_predict['target']=test_y

    trained_models=[]

    for name, model in models:
        print("Fitting model: ", name )
        m = model.fit(train_X, train_y)

        train_predict[name]=m.predict_proba(train_X)[:,1]

        test_predict[name]=m.predict_proba(test_X)[:,1]
        trained_models.append({name: m})

    return train_predict, test_predict, trained_models

def train_test_split(df, date_col, date_format, split_time):
    """
    Provide an train/test split based on a timestamp.
    df = Dataframe (Pandas dataframe).
    date_col = Date column (string).
    date_format = The date format.
    split_time = A specific place to date. (date format)
    """
    split =pd.Timestamp(split_time)
    #Let's convert this to datetime while we are at it.
    df['yrm'] = pd.to_datetime(df[date_col], format=date_format)
    train=df.loc[df['yrm']<split]
    test=df.loc[df['yrm']>split]
    return train, test
