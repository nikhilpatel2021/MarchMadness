# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:19:38 2020

@author: Radin
"""

import pandas as pd
import numpy as np

#Import datasets
end_of_season_stats_df              = pd.DataFrame.from_csv("data_prep_data/end_of_season_stats.csv")
tourney_df                          = pd.DataFrame.from_csv("kaggle_data/MNCAATourneyCompactResults.csv")
seeds_df                            = pd.DataFrame.from_csv("kaggle_data/MNCAATourneySeeds.csv")
teams_df                            = pd.DataFrame.from_csv("kaggle_data/MTeams.csv")
conferences_df                      = pd.DataFrame.from_csv("kaggle_data/MTeamConferences.csv")

#Create 'Season' column for datasets
tourney_df                          = tourney_df.reset_index()
tourney_df['Season']                = tourney_df['Season'].apply(str)
tourney_df['Season']                = tourney_df.Season.str[:4]
tourney_df['Season']                = tourney_df['Season'].astype(int)
tourney_df                          = tourney_df[tourney_df.Season > 2002]
tourney_df                          = tourney_df.reset_index()
tourney_df                          = tourney_df.drop(['index'], axis=1)
seeds_df                            = seeds_df.reset_index()
seeds_df['Season']                  = seeds_df['Season'].apply(str)
seeds_df['Season']                  = seeds_df.Season.str[:4]
seeds_df['Season']                  = seeds_df['Season'].astype(int)
seeds_df                            = seeds_df[seeds_df.Season > 2002]
seeds_df.Seed                       = seeds_df.Seed.apply(str).str[1:3]
conferences_df                      = conferences_df.reset_index()
conferences_df['Season']            = conferences_df['Season'].apply(str)
conferences_df['Season']            = conferences_df.Season.str[:4]
conferences_df['Season']            = conferences_df['Season'].astype(int)
conferences_df                      = conferences_df[conferences_df.Season > 2002]

#Create tournament matchup dataframe containing matchups for every team
tourney_df = tourney_df[['Season', 'WTeamID', 'LTeamID']]
tourney_df['Team1_won'] = 1
tourney_df = tourney_df.rename(index= str, columns={'WTeamID':'Team1_teamID'})
tourney_df = tourney_df.rename(index= str, columns={'LTeamID':'Team2_teamID'})
tourney_matchup_df = pd.DataFrame()

for i in range(0, len(tourney_df)):
    tourney_matchup_df = tourney_matchup_df.append(tourney_df.iloc[i])
    reverse_df = pd.DataFrame({'Season':tourney_df.iloc[i].Season
                               ,'Team1_teamID':tourney_df.iloc[i].Team2_teamID
                               ,'Team2_teamID':tourney_df.iloc[i].Team1_teamID
                               ,'Team1_won' : 0}, index=[0])
    tourney_matchup_df = tourney_matchup_df.append(reverse_df)
    
end_of_season_stats_df_stat_list = list(end_of_season_stats_df)

for stat in list(end_of_season_stats_df):
    end_of_season_stats_df          = end_of_season_stats_df.rename(index = str, columns={stat : 'Team1_' + stat})
end_of_season_stats_df              = end_of_season_stats_df.rename(index = str, columns={'Team1_Season' : 'Season'})
tourney_matchup_df                  = pd.merge(tourney_matchup_df, end_of_season_stats_df, how='left', on= ['Season','Team1_teamID'])

end_of_season_stats_df              = pd.DataFrame.from_csv("data_prep_data/end_of_season_stats.csv")
for stat in list(end_of_season_stats_df):
    end_of_season_stats_df          = end_of_season_stats_df.rename(index = str, columns={stat : 'Team2_' + stat})
end_of_season_stats_df              = end_of_season_stats_df.rename(index = str, columns={'Team2_Season' : 'Season'})
tourney_matchup_df                  = pd.merge(tourney_matchup_df, end_of_season_stats_df, how='left', on= ['Season','Team2_teamID'])

#Add seeds
seeds_df                            = seeds_df.rename(index= str, columns={'TeamID':'Team1_teamID'})
tourney_matchup_df                  = pd.merge(tourney_matchup_df, seeds_df, how= 'left', on= ['Season', 'Team1_teamID'])
seeds_df                            = seeds_df.rename(index= str, columns={'Team1_teamID':'Team2_teamID'})
tourney_matchup_df                  = pd.merge(tourney_matchup_df, seeds_df, how= 'left', on= ['Season', 'Team2_teamID'])

tourney_matchup_df.Seed_x           = tourney_matchup_df.Seed_x.astype(int)
tourney_matchup_df.Seed_y           = tourney_matchup_df.Seed_y.astype(int)
tourney_matchup_df                  = tourney_matchup_df.rename(index= str, columns={'Seed_x':'Team1_seed'})
tourney_matchup_df                  = tourney_matchup_df.rename(index= str, columns={'Seed_y':'Team2_seed'})

#Add differences in stats as a stat
end_of_season_stats_df_stat_list.append('seed')
for stat in end_of_season_stats_df_stat_list[2:len(end_of_season_stats_df_stat_list)]:
    tourney_matchup_df['diff_' + stat] = tourney_matchup_df['Team1_' + stat] - tourney_matchup_df['Team2_' + stat]

#Add conferences
conferences_df                      = conferences_df.rename(index= str, columns={'TeamID':'Team1_teamID'})
tourney_matchup_df                  = pd.merge(tourney_matchup_df, conferences_df, how= 'left', on= ['Season', 'Team1_teamID'])
conferences_df                      = conferences_df.rename(index= str, columns={'Team1_teamID':'Team2_teamID'})
tourney_matchup_df                  = pd.merge(tourney_matchup_df, conferences_df, how= 'left', on= ['Season', 'Team2_teamID'])

#Remove first 4 games (16v16, 11v11)
tourney_matchup_df = tourney_matchup_df[~(((tourney_matchup_df.Team1_seed == 16) |(tourney_matchup_df.Team1_seed == 11)) & (tourney_matchup_df.diff_seed == 0))]

#Get dummy variables for conferences
tourney_matchup_df                      = pd.get_dummies(tourney_matchup_df)

tourney_matchup_df.to_csv("data_prep_data/tourney_matchup_df.csv")
