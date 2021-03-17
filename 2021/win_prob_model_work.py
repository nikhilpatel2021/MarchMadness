# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 18:29:47 2021

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
tourney_matchup_df = pd.read_csv("created_data/tourney_matchup_df.csv")

stats = list(tourney_matchup_df.columns[6:])
target = 'Team1_won'
cols_to_remove = [
    'Team1_team_W',
     'Team1_team_L',
     'Team1_team_G',
     'Team2_team_W',
     'Team2_team_L',
     'Team2_team_G',
     'diff_team_L',
     'diff_team_G'
     ]
for col in cols_to_remove:
    stats.remove(col)
kf = KFold(n_splits=5, shuffle=True)
X = np.array(tourney_matchup_df[stats])
y = np.array(tourney_matchup_df[target])
scaler_X = StandardScaler()
scaler_X.fit(X)
X_scaled = scaler_X.transform(X)
lasso_results = []
alphas = [0, 0.0001, 0.001, 0.01, 0.1, 1]
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    alpha_results = {}
    for alpha in alphas:
        lasso = Lasso(alpha=alpha).fit(X_train, y_train)
        preds_train = lasso.predict(X_train)
        preds_test = lasso.predict(X_test)
        alpha_results[alpha] = roc_auc_score(y_test, preds_test)
    lasso_results.append(alpha_results)
alpha_dict = {}
for alpha in alphas:
    alpha_dict[alpha] = []
    for result in lasso_results:
        alpha_dict[alpha].append(result[alpha])
    print(alpha, np.mean(alpha_dict[alpha]))
    
lasso_001 = Lasso(alpha=0.01).fit(X_scaled, y)
coefficients = pd.DataFrame(lasso_001.coef_, stats, columns=(['0.01']))
lasso_0001 = Lasso(alpha=0.001).fit(X_scaled, y)
coefficients['0.001'] = lasso_0001.coef_

clf = LogisticRegressionCV(cv=5, random_state=0, penalty='l1', solver='saga', scoring='roc_auc').fit(X_scaled, y)
coefficients['logistic'] = pd.DataFrame(clf.coef_.T, stats)
scores = clf.scores_
avg_scores = np.mean(scores[1], axis=0)

###############################################################################
kf = KFold(n_splits=5, shuffle=True)
X = np.array(tourney_matchup_df[stats])
y = np.array(tourney_matchup_df[target])
inc = 0
tot = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = LogisticRegressionCV(cv=5, random_state=0, penalty='l1', solver='saga', scoring='roc_auc').fit(X_train, y_train)
    preds_test = clf.predict(X_test)
    preds_test[preds_test > 0.5] = 1
    preds_test[preds_test < 0.5] = 0
    inc += sum(abs(y_test - preds_test))
    tot += len(y_test)
analysis_df = pd.DataFrame(y_test)
analysis_df['pred'] = 0
analysis_df['pred'][preds_test>0.5] = 1

chalk = tourney_matchup_df[tourney_matchup_df.diff_seed != 0]
chalk['chalk'] = 0
chalk['chalk'][chalk.diff_seed < 0] = 1
print("Chalk games correct: ", len(chalk)-sum(abs(chalk.Team1_won - chalk.chalk)), " out of: ", len(chalk))
print("Chalk correct pct: ", 1 - sum(abs(chalk.Team1_won - chalk.chalk)) / len(chalk))
print("New model games correct: ", len(analysis_df)-sum(abs(analysis_df.Team1_won - analysis_df.pred)), " out of: ", len(analysis_df))
print("New model correct pct: ", 1 - sum(abs(analysis_df.Team1_won - analysis_df.pred)) / len(analysis_df))
