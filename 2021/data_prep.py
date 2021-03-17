# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:32:06 2020

@author: Radin
"""

import pandas as pd
import numpy as np
import os
import sys
directory = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(directory, '../kaggle_data/MRegularSeasonDetailedResults.csv'))

# Read in detailed regular season game data
regular_season_df = pd.read_csv("kaggle_data_2021/MRegularSeasonDetailedResults.csv/MRegularSeasonDetailedResults.csv")

# Add possessions as a stat each game for both teams
for x in ['W', 'L']:
    regular_season_df[x + 'Poss'] = regular_season_df[x+'FGA'] \
                                         + regular_season_df[x+'TO'] \
                                         + regular_season_df[x+'FTA'] * 0.44 \
                                         - regular_season_df[x+'OR']
    
# Begin creating end-of-season-stats for each team for each season  
# These stats will be a collection of simple team stats, advanced team stats,
# and advanced team opponent stats
w_stats = [col for col in regular_season_df.columns if col[0] == 'W' and col not in ['WLoc','WTeamID']]
l_stats = [col for col in regular_season_df.columns if col[0] == 'L' and col not in ['LTeamID']]

w_team_stats = regular_season_df.groupby(['Season', 'WTeamID'])[w_stats].agg(sum)
w_team_stats['team_W'] = regular_season_df.groupby(['Season', 'WTeamID'])['WTeamID'].agg('count')

w_oppo_stats = regular_season_df.groupby(['Season', 'WTeamID'])[l_stats].agg(sum)

l_team_stats = regular_season_df.groupby(['Season', 'LTeamID'])[l_stats].agg(sum)
l_team_stats['team_L'] = regular_season_df.groupby(['Season', 'LTeamID'])['LTeamID'].agg('count')

l_oppo_stats = regular_season_df.groupby(['Season', 'LTeamID'])[w_stats].agg(sum)
w_team_stats.reset_index(inplace=True);l_team_stats.reset_index(inplace=True);w_oppo_stats.reset_index(inplace=True);l_oppo_stats.reset_index(inplace=True);

df = pd.merge(w_team_stats,l_team_stats,how='outer',left_on=['Season','WTeamID'],right_on=['Season','LTeamID'])
df2 = pd.merge(w_oppo_stats,l_oppo_stats,how='outer',left_on=['Season','WTeamID'],right_on=['Season','LTeamID'])
for df_ in [df, df2]: 
    df_['teamID'] = df_['WTeamID'].fillna(df_['LTeamID'])
    df_.fillna(0, inplace=True)
    df_.drop(columns=['WTeamID', 'LTeamID'], inplace=True)
del regular_season_df, w_team_stats, w_oppo_stats, l_team_stats, l_oppo_stats, df_

# Calculate stats
df['team_G'] = df['team_W'] + df['team_L']
df['team_Wpct'] = df['team_W'] / df['team_G']
for col in w_stats:
    df['team_' + col[1:] + '_PG'] = (df[col] + df['L'+col[1:]]) / df['team_G']
    df['team_' + col[1:] + '_PP'] = (df[col] + df['L'+col[1:]]) / (df['WPoss'] + df['LPoss']) 
df['team_Drtg'] = (df2['WScore'] + df2['LScore']) / (df2['WPoss'] + df2['LPoss']) 
for t in ['team_', 'oppo_']:
    if t == 'team_':
        df_1 = df
        df_2 = df2
        
    else:
        df_1 = df2
        df_2 = df
    df[t + 'FTr'] = (df_1['WFTA'] + df_1['LFTA']) / (df_1['WFGA'] + df_1['LFGA'])
    df[t + '3PAr'] = (df_1['WFGA3'] + df_1['LFGA3']) / (df_1['WFGA'] + df_1['LFGA'])
    df[t + 'TSpct'] = (df_1['WScore'] + df_1['LScore']) / (2*(df_1['WFGA'] + df_1['LFGA']) + 0.44*(df_1['WFTA'] + df_1['LFTA']))
    df[t + 'TRBpct'] = (df_1['WOR'] + df_1['LOR'] + df_1['WDR'] + df_1['LDR']) / (df_1['WOR'] + df_1['LOR'] + df_1['WDR'] + df_1['LDR'] + df_2['WOR'] + df_2['LOR'] + df_2['WDR'] + df_2['LDR'])
    df[t + 'ASTpct'] = (df_1['WAst'] + df_1['LAst']) / (df_1['WFGM'] + df_1['LFGM'])
    df[t + 'STLpct'] = (df_1['WStl'] + df_1['LStl']) / (df_2['WPoss'] + df_2['LPoss'])
    df[t + 'BLKpct'] = (df_1['WBlk'] + df_1['LBlk']) / (df_2['WFGA'] + df_2['LFGA'] - df_2['WFGA3'] + df_2['LFGA3'])
    df[t + 'eFGpct'] = (df_1['WFGM'] + df_1['LFGM'] + 0.5*df_1['WFGM3'] + 0.5*df_1['LFGM3']) / (df_1['WFGA'] + df_1['LFGA'])
    df[t + 'TOVpct'] = (df_1['WTO'] + df_1['LTO']) / (df_1['WPoss'] + df_1['LPoss'])
    df[t + 'ORBpct'] = (df_1['WOR'] + df_1['LOR']) / (df_1['WOR'] + df_1['LOR'] + df_2['WDR'] + df_2['LDR'])
    df[t + 'FTperFG'] = (df_1['WFTM'] + df_1['LFTM']) / (df_1['WFGA'] + df_1['LFGA'])
del df_1, df_2, df2

# Delete some stats
w_stats.extend(l_stats)
w_stats.append('team_Poss_PP')
df.drop(columns=w_stats, inplace=True)
df.to_csv("created_data/end_of_season_stats.csv", index=False)
