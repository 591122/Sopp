import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score

def randomforest(arg):
    
    X = arg.drop(columns=['class'])
    y = arg['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    # Import and train a machine learning model
    rf = RandomForestClassifier( n_estimators=100)
    rf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = rf.predict(X_test)

    # Evaluate the model
    score = accuracy_score(y_test, y_pred)
    return score


def gradientboost(arg):
    X = arg.drop(columns=['class'])
    y = arg['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    # Import and train a machine learning model
    rf = GradientBoostingClassifier( random_state=0, n_estimators=100)
    rf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = rf.predict(X_test)

    # Evaluate the model
    score = accuracy_score(y_test, y_pred)
    return score

def linearregression(arg):
    X = arg.drop(columns=['class'])
    y = arg['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    # Import and train a machine learning model
    rf = LinearRegression()
    rf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = rf.predict(X_test)

    # Evaluate the model
    score = rf.score(X, y, sample_weight=None)
    return score

def decisionregressor(arg):
    X = arg.drop(columns=['class'])
    y = arg['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    # Import and train a machine learning model
    rf = DecisionTreeRegressor()
    rf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = rf.predict(X_test)

    # Evaluate the model
    score = rf.score(X, y, sample_weight=None)
    return score

def mlpregression(arg):
    X = arg.drop(columns=['class'])
    y = arg['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    # Import and train a machine learning model
    rf = MLPRegressor(max_iter=500)
    rf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = rf.predict(X_test)

    # Evaluate the model
    score = rf.score(X, y, sample_weight=None)
    return score

def supportvectoregressor(arg):
    X = arg.drop(columns=['class'])
    y = arg['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    regr.fit(X, y)
    # Evaluate the model
    score = regr.score(X, y, sample_weight=None)
    return score

def kneighborsregressor(arg):
    X = arg.drop(columns=['class'])
    y = arg['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


    # Import and train a machine learning model
    rf = KNeighborsRegressor()
    rf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = rf.predict(X_test)

    # Evaluate the model
    score = rf.score(X, y, sample_weight=None)
    return score