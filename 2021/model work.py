# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:27:10 2021

@author: Radin
"""
import pandas as pd
import numpy as np
import random
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegressionCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
random.seed(1)
#end_of_season_stats_df = pd.read_csv("new_data/end_of_season_stats.csv")
tourney_matchup_df = pd.read_csv("new_data/tourney_matchup_df.csv")
"""
# PtDiff model work
stats = list(tourney_matchup_df.columns[6:])
target = 'PtDiff'
X = tourney_matchup_df[stats]
y = tourney_matchup_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Linear regression
alphas = [0, 0.0001, 0.001, 0.01, 0.1, 1]
results = {}
for alpha in alphas:
    lasso = Lasso(alpha=alpha).fit(X_train, y_train)
    preds_train = lasso.predict(X_train)
    preds_test = lasso.predict(X_test)
    print('Alpha',alpha)
    print("Train MSE: ", mean_squared_error(y_train, preds_train, squared=False))
    print("Test MSE: ", mean_squared_error(y_test, preds_test, squared=False))

lasso = Lasso(alpha=.1).fit(X_train, y_train)
coefficients = pd.DataFrame(lasso.coef_, stats)
plt.scatter(y_test, preds_test)
plt.plot([-30,30],[-30,30])
analysis_df = pd.DataFrame(y_test)
analysis_df['pred'] = preds_test
analysis_df['diff'] = abs(analysis_df['PtDiff'] - analysis_df['pred'])
analysis_df['correct'] = (analysis_df['PtDiff'] * analysis_df['pred']) / (abs(analysis_df['PtDiff'] * analysis_df['pred']))

# Neural net
tuples_list = [
    (2,5,3)
    ,(1)
    ,(2)
    ,(5)
    ,(10,5)
    ]
mlp_results = {}
for t in tuples_list:
    regr = MLPRegressor(t, random_state=1, max_iter=500).fit(X_train, y_train)
    preds_train = regr.predict(X_train)
    preds_test = regr.predict(X_test)
    train_mse = mean_squared_error(y_train, preds_train, squared=False)
    test_mse = mean_squared_error(y_test, preds_test, squared=False)
    mlp_results[t] = [train_mse, test_mse]
"""

### Lasso with .1 is best ###

# Team1_won work
stats = list(tourney_matchup_df.columns[6:])
target = 'Team1_won'
# year = 2019
# X_train = tourney_matchup_df[tourney_matchup_df.Season < year][stats]
# y_train = tourney_matchup_df[tourney_matchup_df.Season < year][target]
# X_test = tourney_matchup_df[tourney_matchup_df.Season == year][stats]
# y_test = tourney_matchup_df[tourney_matchup_df.Season == year][target]
X = tourney_matchup_df[stats]
y = tourney_matchup_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Linear regression
alphas = [0, 0.0001, 0.001, 0.01, 0.1, 1]
results = {}
for alpha in alphas:
    lasso = Lasso(alpha=alpha).fit(X_train, y_train)
    preds_train = lasso.predict(X_train)
    preds_test = lasso.predict(X_test)
    print('Alpha',alpha)
    print("Train AUC: ", roc_auc_score(y_train, preds_train))
    print("Test AUC: ", roc_auc_score(y_test, preds_test))

lasso = Lasso(alpha=0.01).fit(X_train, y_train)
coefficients = pd.DataFrame(lasso.coef_, stats)
plt.scatter(y_test, preds_test)
analysis_df = pd.DataFrame(y_test)
analysis_df['pred'] = 0
analysis_df['pred'][preds_test>0.5] = 1
1 - sum(abs(analysis_df.Team1_won - analysis_df.pred)) / len(analysis_df)

# Logistic
clf = LogisticRegressionCV(cv=5, random_state=0, penalty='l2', scoring='roc_auc').fit(X_train, y_train)
coefficients = pd.DataFrame(clf.coef_.T, stats)
preds_train = clf.predict(X_train)
preds_test = clf.predict(X_test)
print("Train AUC: ", roc_auc_score(y_train, preds_train))
print("Test AUC: ", roc_auc_score(y_test, preds_test))
analysis_df = pd.DataFrame(y_test)
analysis_df['pred'] = 0
analysis_df['pred'][preds_test>0.5] = 1
1 - sum(abs(analysis_df.Team1_won - analysis_df.pred)) / len(analysis_df)

# Neural net
# tuples_list = [
#     (2,5,3)
#     ,(1)
#     ,(2)
#     ,(5)
#     ,(10,5)
#     ]
# mlp_results2 = {}
# for t in tuples_list:
#     regr = MLPRegressor(t, random_state=1, max_iter=500).fit(X_train, y_train)
#     preds_train = regr.predict(X_train)
#     preds_test = regr.predict(X_test)
#     train_mse = roc_auc_score(y_train, preds_train)
#     test_mse = roc_auc_score(y_test, preds_test)
#     mlp_results2[t] = [train_mse, test_mse]
### use clf ###

"""
# PtTotal model work
stats = list(tourney_matchup_df.columns[6:])
target = 'PtTotal'
X = tourney_matchup_df[stats]
y = tourney_matchup_df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Linear regression
alphas = [0, 0.0001, 0.001, 0.01, 0.1, 1]
results = {}
for alpha in alphas:
    lasso = Lasso(alpha=alpha).fit(X_train, y_train)
    preds_train = lasso.predict(X_train)
    preds_test = lasso.predict(X_test)
    print('Alpha',alpha)
    print("Train MSE: ", mean_squared_error(y_train, preds_train, squared=False))
    print("Test MSE: ", mean_squared_error(y_test, preds_test, squared=False))

lasso = Lasso(alpha=0.1).fit(X_train, y_train)
coefficients = pd.DataFrame(lasso.coef_, stats)
plt.scatter(y_test, preds_test)
plt.plot([-30,30],[-30,30])
analysis_df = pd.DataFrame(y_test)
analysis_df['pred'] = preds_test
analysis_df['diff'] = abs(analysis_df[target] - analysis_df['pred'])
analysis_df['correct'] = (analysis_df[target] > analysis_df['pred'])

# Neural net
tuples_list = [
    (5,5,5)
    ,(10,5,5)
    ,(5,10,5)
    ,(4,3,2)
    ,(10,5,2)
    ]
mlp_results = {}
for t in tuples_list:
    regr = MLPRegressor(t, random_state=1, max_iter=500).fit(X_train, y_train)
    preds_train = regr.predict(X_train)
    preds_test = regr.predict(X_test)
    train_mse = mean_squared_error(y_train, preds_train, squared=False)
    test_mse = mean_squared_error(y_test, preds_test, squared=False)
    mlp_results[t] = [train_mse, test_mse]
"""
"""
chalk = tourney_matchup_df[tourney_matchup_df.diff_seed != 0]
chalk['chalk'] = 0
chalk['chalk'][chalk.diff_seed < 0] = 1
print("Chalk games correct: ", len(chalk)-sum(abs(chalk.Team1_won - chalk.chalk)), " out of: ", len(chalk))
print("Chalk correct pct: ", 1 - sum(abs(chalk.Team1_won - chalk.chalk)) / len(chalk))
print("New model games correct: ", len(analysis_df)-sum(abs(analysis_df.Team1_won - analysis_df.pred)), " out of: ", len(analysis_df))
print("New model correct pct: ", 1 - sum(abs(analysis_df.Team1_won - analysis_df.pred)) / len(analysis_df))
"""
"""
pct = []
for i in range(round(max(analysis_df['diff']))):
    pct.append(sum(analysis_df['diff'] < i) / len(analysis_df))
    
vegas_spreads2018 = pd.read_csv("C:/Users/Radin/Downloads/spread_diffs.csv")
vegas_spread = []
for i in range(round(max(vegas_spreads2018['diff']))):
    vegas_spread.append(sum(vegas_spreads2018['diff'] < i) / len(vegas_spreads2018))
    
plt.plot(pct)
plt.plot(vegas_spread)
plt.legend(['Me', 'Vegas'], loc='upper left')
plt.show()

vegas_opens = vegas_spreads2018[vegas_spreads2018.Open != "NL"]
vegas_opens.loc[vegas_opens.Open == "pk", 'Open'] = 0
vegas_opens['Open'] = vegas_opens.Open.astype(float)
vegas_opens = vegas_opens[vegas_opens['Open'] < 50]
vegas_opens['diff_open'] = abs(vegas_opens.finalspread - vegas_opens.Open)

vegas_spread_opens = []
for i in range(round(max(vegas_opens['diff_open']))):
    vegas_spread_opens.append(sum(vegas_opens['diff_open'] < i) / len(vegas_opens))
    
plt.plot(pct)
plt.plot(vegas_spread)
plt.plot(vegas_spread_opens)
plt.legend(['Me', 'Vegas Close', 'Vegas Open'], loc='upper left')
plt.show()

# Covering
print("my predictions cover my spread: ", sum(analysis_df.PtDiff > analysis_df.pred) / len(analysis_df))
print("vegas open cover the spread: ",sum(vegas_opens.Open > vegas_opens.finalspread) / len(vegas_opens))
print("vegas close cover the spread: ",sum(vegas_spreads2018.Close > vegas_spreads2018.finalspread) / len(vegas_spreads2018))

print("my predictions dont cover my spread: ", sum(analysis_df.PtDiff < analysis_df.pred) / len(analysis_df))
print("vegas open dont cover the spread: ",sum(vegas_opens.Open < vegas_opens.finalspread) / len(vegas_opens))
print("vegas close dont cover the spread: ",sum(vegas_spreads2018.Close < vegas_spreads2018.finalspread) / len(vegas_spreads2018))
"""