# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 21:20:29 2021

@author: Radin
"""
import pandas as pd

df = pd.read_csv("C:/Users/Radin/Downloads/ncaa_2018_2019.csv")
new_df = pd.DataFrame()
cols = [3,6,7,8,9]
new_df['Date'] = df.iloc[::2, 0].reset_index(drop=True)
for col in cols:
    V = df.iloc[::2, col].reset_index(drop=True)
    H = df.iloc[1::2, col].reset_index(drop=True)
    new_df = pd.concat([new_df,V,H], axis=1)
del df, H, V
new_df.columns = ['Date', 'Visitor', 'Home', 'VisitorFinal', 'HomeFinal', 'Open1', 'Open2', 'Close1', 'Close2',
       'VisitorML', 'HomeML']
for col in ['Open1', 'Open2', 'Close1', 'Close2','VisitorML', 'HomeML']:
    new_df = new_df[new_df[col] != "NL"]
    new_df.loc[new_df[col] == "pk", col] = 0
    new_df[col] = new_df[col].astype(float)
# Remember to handle 0 in the spread
new_df['OpenTotal'] = new_df['Open1'][new_df['Open1'] > new_df['Open2']]
new_df['OpenTotal'] = new_df['OpenTotal'].fillna(new_df['Open2'])
new_df['CloseTotal'] = new_df['Close1'][new_df['Close1'] > new_df['Close2']]
new_df['CloseTotal'] = new_df['CloseTotal'].fillna(new_df['Close2'])
new_df['OpenSpread'] = new_df['Open1'][new_df['Open1'] < new_df['Open2']]
new_df['OpenSpread'] = new_df['OpenSpread'].fillna(new_df['Open2'])
new_df['CloseSpread'] = new_df['Close1'][new_df['Close1'] < new_df['Close2']]
new_df['CloseSpread'] = new_df['CloseSpread'].fillna(new_df['Close2'])
new_df['OpenHomeFave'] = new_df['Open1'] > new_df['Open2']
new_df['CloseHomeFave'] = new_df['Close1'] > new_df['Close2']

new_df['SpreadCoverFaveOpen'] = new_df['HomeFinal'][new_df['OpenHomeFave']] - new_df['VisitorFinal'][new_df['OpenHomeFave']] > new_df['OpenSpread'][new_df['OpenHomeFave']]
new_df.loc[~new_df['OpenHomeFave'], 'SpreadCoverFaveOpen'] = new_df['VisitorFinal'][~new_df['OpenHomeFave']] - new_df['HomeFinal'][~new_df['OpenHomeFave']] > new_df['OpenSpread'][~new_df['OpenHomeFave']]

new_df['SpreadCoverFaveClose'] = new_df['HomeFinal'][new_df['CloseHomeFave']] - new_df['VisitorFinal'][new_df['CloseHomeFave']] > new_df['CloseSpread'][new_df['CloseHomeFave']]
new_df.loc[~new_df['CloseHomeFave'], 'SpreadCoverFaveClose'] = new_df['VisitorFinal'][~new_df['CloseHomeFave']] - new_df['HomeFinal'][~new_df['CloseHomeFave']] > new_df['CloseSpread'][~new_df['CloseHomeFave']]


new_df['SpreadDiffOpen'] = new_df['HomeFinal'][new_df['OpenHomeFave']] - new_df['VisitorFinal'][new_df['OpenHomeFave']] - new_df['OpenSpread'][new_df['OpenHomeFave']]
new_df.loc[~new_df['OpenHomeFave'], 'SpreadDiffOpen'] = new_df['VisitorFinal'][~new_df['OpenHomeFave']] - new_df['HomeFinal'][~new_df['OpenHomeFave']] - new_df['OpenSpread'][~new_df['OpenHomeFave']]
new_df['SpreadDiffClose'] = new_df['HomeFinal'][new_df['CloseHomeFave']] - new_df['VisitorFinal'][new_df['CloseHomeFave']] - new_df['CloseSpread'][new_df['CloseHomeFave']]
new_df.loc[~new_df['CloseHomeFave'], 'SpreadDiffClose'] = new_df['VisitorFinal'][~new_df['CloseHomeFave']] - new_df['HomeFinal'][~new_df['CloseHomeFave']] - new_df['CloseSpread'][~new_df['CloseHomeFave']]

new_df['TotalFinal'] = new_df['HomeFinal'] + new_df['VisitorFinal']
new_df['TotalOverOpen'] = new_df['TotalFinal'] > new_df['OpenTotal']
new_df['TotalOverClose'] = new_df['TotalFinal'] > new_df['CloseTotal']
new_df['TotalFinalDiffOpen'] = new_df['TotalFinal'] - new_df['OpenTotal']
new_df['TotalFinalDiffClose'] = new_df['TotalFinal'] - new_df['CloseTotal']

# No favorite in pickem game
new_df.loc[new_df['OpenSpread'] == 0, 'SpreadCoverFaveOpen'] = None
new_df.loc[new_df['CloseSpread'] == 0, 'SpreadCoverFaveClose'] = None
# Handle pushes
new_df.loc[new_df['SpreadDiffOpen'] == 0, 'SpreadCoverFaveOpen'] = None
new_df.loc[new_df['SpreadDiffClose'] == 0, 'SpreadCoverFaveClose'] = None
new_df.loc[new_df['TotalFinalDiffOpen'] == 0, 'TotalOverOpen'] = None
new_df.loc[new_df['TotalFinalDiffClose'] == 0, 'TotalOverClose'] = None


tourney_matchup_df = pd.read_csv("new_data/tourney_matchup_df.csv")
teams_df = pd.read_csv("../kaggle_data/MTeams.csv").loc[:, ['TeamID', 'TeamName']]
tourney_matchup_df = tourney_matchup_df[tourney_matchup_df.Season == 2019]
tourney_matchup_df = pd.merge(tourney_matchup_df, teams_df, how='left', left_on='Team1_teamID', right_on='TeamID')
tourney_matchup_df = pd.merge(tourney_matchup_df, teams_df, how='left', left_on='Team2_teamID', right_on='TeamID')
march_bets = new_df[(new_df.Date > 300) & (new_df.Date < 500)]
tourney_matchup_df2 = pd.merge(tourney_matchup_df, new_df, how='left', left_on=['TeamName_x', 'TeamName_y'], right_on=['Home','Visitor'])
