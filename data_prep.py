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
sys.path.append(os.path.join(directory, 'kaggle_data/MRegularSeasonDetailedResults.csv'))

regular_season_df = pd.read_csv("kaggle_data/MRegularSeasonDetailedResults.csv/MRegularSeasonDetailedResults.csv")
regular_season_df['WPoss']  = regular_season_df['WFGA'] \
                             + regular_season_df['WTO'] \
                             + regular_season_df['WFTA'] * 0.44 \
                             - regular_season_df['WOR']
                            
regular_season_df['LPoss']  = regular_season_df['LFGA'] \
                             + regular_season_df['LTO'] \
                             + regular_season_df['LFTA'] * 0.44\
                             - regular_season_df['LOR']