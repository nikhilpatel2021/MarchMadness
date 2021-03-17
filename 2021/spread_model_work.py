# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:05:29 2021

@author: Radin
"""
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
tourney_matchup_df = pd.read_csv("new_data/tourney_matchup_df.csv")

stats = list(tourney_matchup_df.columns[6:])
target = 'PtDiff'
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
kf = KFold(n_splits=5)
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
        alpha_results[alpha] = mean_squared_error(y_test, preds_test, squared=False)
        win_preds = (preds_test - min(preds_test)) / (max(preds_test) - min(preds_test))
        y_wins = np.array(tourney_matchup_df.Team1_won)[test_index]
        alpha_results[alpha] = roc_auc_score(y_wins, win_preds)
    lasso_results.append(alpha_results)
alpha_dict = {}
for alpha in alphas:
    alpha_dict[alpha] = []
    for result in lasso_results:
        alpha_dict[alpha].append(result[alpha])
    print(alpha, np.mean(alpha_dict[alpha]))
    
lasso_01 = Lasso(alpha=0.1).fit(X_scaled, y)
coefficients = pd.DataFrame(lasso_01.coef_, stats, columns=(['0.1']))
lasso_001 = Lasso(alpha=0.01).fit(X_scaled, y)
coefficients['0.01'] = lasso_001.coef_
