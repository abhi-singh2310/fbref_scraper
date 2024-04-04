# Project: FbRef Web Scraper
# Author: Abhijot Singh
# Date: 03/01/2024

# File: scraper.py
# Purpose: Contains the set of functions used to scrape custom data sources from the fbref website.

import pandas as pd
import numpy as np
import requests
from IPython.display import display

import thefuzz
from thefuzz import process, fuzz

import streamlit as st
import streamlit_pandas as sp

from datetime import timedelta
from ratelimit import limits, sleep_and_retry
import requests

from constants import *

# Headers are needed for the request url call to be valid.
headers = {
    "Connection": "keep-alive",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36"
}

# From the scraped data, remove duplicate values. Duplicates occur when a player has transferred teams.
# The data point with the most 90s completed is kept and the other is removed.
# For example, Neil Maupay's Brentford stats are kept and Everton removed.
def data_filtering(df_input):
    df = df_input.copy()
    df['90s'] = pd.to_numeric(df['90s'], errors='coerce')
    df = df.sort_values(by='90s', ascending=False)
    df.drop_duplicates(subset='Player', keep='first', inplace=True)
    df = df.loc[df['Player'] != 'Player']
    # Reorder the data by first name ascending (commented line sorts by last name)
    df = df.sort_values(by='Player', ascending=True)
    # df = df.loc[df['Player'].apply(lambda x: x.split()[-1]).sort_values().index]
    df = df.reset_index(drop=True)
    return df

# Generate all necessary urls per league and metric
def create_link(league, metric):
    if league == 'Premier League':
        url = "https://fbref.com/en/comps/9/" + metric + '/' + PREM
    elif league == 'Bundesliga':
        url = "https://fbref.com/en/comps/20/" + metric + '/' + BUND
    elif league == 'La Liga':
        url = "https://fbref.com/en/comps/12/" + metric + '/' + LALIGA
    elif league == 'Serie A':
        url = "https://fbref.com/en/comps/11/" + metric + '/' + SERIEA
    elif league == 'Ligue 1':
        url = "https://fbref.com/en/comps/13/" + metric + '/' + LIGUE_1
    elif league == 'Top 5 Europe':
        url = "https://fbref.com/en/comps/Big5/" + metric + '/players/' + ALL
        
    return url

# Function requires limits to url requests being made. Only 5 can be made per 60 seconds to prevent a 429 error and blocking.
@sleep_and_retry
@limits(calls=5, period=timedelta(seconds=60).total_seconds())
# Extract the relevant player dataset from the fbref url
def player_extract(url, league):
    # Extract data from the url
    res = requests.get(url, headers=headers).text
    htmlStr = res.replace('<!--', '')
    htmlStr = htmlStr.replace('-->', '')

    dfs = pd.read_html(htmlStr, header=1)
    if league == 'Top 5 Europe':
        try:
            player_table = dfs[1]
        except:
            player_table = dfs[0]
    else:
        player_table = dfs[2]

    return player_table

@sleep_and_retry
@limits(calls=5, period=timedelta(seconds=60).total_seconds())
# Extract the team for dataset from the fbref url
def team_extract_for(url, league):
    # Extract data from the url
    res = requests.get(url, headers=headers).text
    htmlStr = res.replace('<!--', '')
    htmlStr = htmlStr.replace('-->', '')

    dfs = pd.read_html(htmlStr, header=1)
    if league == 'Top 5 Europe':
        try:
            team_table = dfs[1]
        except:
            team_table = dfs[0]
    else:
        team_table = dfs[0]

    return team_table

@sleep_and_retry
@limits(calls=5, period=timedelta(seconds=60).total_seconds())
# Extract all Squads in Top 5 Leagues
def squad_extract(url):
    # Extract data from the url
    res = requests.get(url, headers=headers).text
    htmlStr = res.replace('<!--', '')
    htmlStr = htmlStr.replace('-->', '')

    dfs = pd.read_html(htmlStr)
    squad_table = dfs[0]

    return squad_table

# Fuzzy Match the correct squad name to downloaded position data
def fuzz_squad(df_dl, df_output):
    df_output['Player_Squad'] = df_output['Player'].astype('string') + df_output['Team'].astype('string')

    url = 'https://fbref.com/en/comps/Big5/Big-5-European-Leagues-Stats'
    df_fbref = squad_extract(url)
    # Get set of teams in top 5 european leagues
    fbref_teams = pd.DataFrame(df_fbref['Squad'].unique())
    fbref_teams.rename(columns={0:'Squad'},inplace=True)
    # Apply fuzzy match to downloaded dataset - Filter to top 5 leagues
    df_fixed = df_dl.loc[df_dl['league'].isin(['Premier League','Serie A','Bundesliga','LaLiga','Ligue 1'])]
    df_fixed.rename(columns={'team_name':'Squad'}, inplace=True)

    df_fixed['Squad'] = df_fixed["Squad"].apply(
    lambda x: process.extractOne(x, fbref_teams["Squad"], scorer=fuzz.partial_ratio)[0]
    )

    df_fixed['Player_Squad'] = df_fixed['player_name'].astype('string') + df_fixed['Squad'].astype('string')
    df_output = df_output.merge(df_fixed[['player_pos', 'Player_Squad']], on='Player_Squad', how='left')

    df_output = df_output.drop(columns=['Player_Squad'])
    df_output.rename(columns={'player_pos':'Role'}, inplace=True)

    pop_col = df_output.pop('Position')
    df_output.insert(2,"Position", pop_col)
    pop_col = df_output.pop('Role')
    df_output.insert(3,"Role", pop_col)

    return df_output

# This will cache the data in streamlit so it does not need to be re-run each time an action is made.
@st.cache_data(experimental_allow_widgets=True)
# Extract all relevant data sources for the metrics to be created
def table_extracts(league):
    # Overall Stats
    metric = 'stats'
    stat_url = create_link(league, metric)
    print(stat_url)
    df_stat_out = player_extract(stat_url, league)
    # Order by GP and remove player duplicates, then re-order by alphabet
    df_stat = data_filtering(df_stat_out)
    # GCA
    metric = 'gca'
    gca_url = create_link(league, metric)
    df_gca_out = player_extract(gca_url, league)
    df_gca = data_filtering(df_gca_out)
    # Playing Time
    metric = 'playingtime'
    pt_url = create_link(league, metric)
    df_playingtime_out = player_extract(pt_url, league)
    df_playingtime = data_filtering(df_playingtime_out)
    # Defense
    metric = 'defense'
    def_url = create_link(league, metric)
    df_def_out = player_extract(def_url, league)
    df_def = data_filtering(df_def_out)
    # Passing
    metric = 'passing'
    pass_url = create_link(league, metric)
    df_pass_out = player_extract(pass_url, league)
    df_pass = data_filtering(df_pass_out)
    # Shooting
    metric = 'shooting'
    shoot_url = create_link(league, metric)
    df_shoot_out = player_extract(shoot_url, league)
    df_shoot = data_filtering(df_shoot_out)
    # Possession
    metric = 'possession'
    pos_url = create_link(league, metric)
    df_pos_out = player_extract(pos_url, league)
    df_pos = data_filtering(df_pos_out)

    if league == 'Top 5 Europe':
        eur_pos = 'https://fbref.com/en/comps/Big5/possession/squads/Big-5-European-Leagues-Stats'
        df_team_pos = team_extract_for(eur_pos, league)
    else:
        df_team_pos = team_extract_for(pos_url, league)
        df_team_pos = df_team_pos.loc[df_team_pos['Squad'] != 'Squad']
        df_team_pos = df_team_pos.reset_index(drop=True)
    # Misc
    metric = 'misc'
    misc_url = create_link(league, metric)
    df_misc_out = player_extract(misc_url, league)
    df_misc = data_filtering(df_misc_out)
    # Goalkeeper
    metric = 'keepers'
    keepers_url = create_link(league, metric)
    df_keeper_out = player_extract(keepers_url, league)
    df_keeper = data_filtering(df_keeper_out)
    # Goalkeeper Advanced
    metric = 'keepersadv'
    adv_url = create_link(league, metric)
    df_keepadv_out = player_extract(adv_url, league)
    df_keepadv = data_filtering(df_keepadv_out)
    
    return df_stat, df_gca, df_playingtime, df_def, df_pass, df_shoot, df_pos, df_team_pos, df_misc, df_keeper, df_keepadv

### METRICS ###
# Each metric requires tables from the table extracts above and can be adjusted if user has knowledge of the fbref base datasets.

def overall_perf_att(df_stat, df_gca):
    # Filter to necessary columns
    df_stat = df_stat[['Player','Pos','Squad','MP', 'Gls', 'Ast', 'xG+xAG']]
    df_gca = df_gca[['Player','SCA','SCA90']]

    df_output = df_stat.merge(df_gca, on='Player', how='left')

    df_output = df_output[['Player','Pos','Squad','MP', 'Gls', 'Ast', 'xG+xAG', 'SCA','SCA90']]
    df_output.rename(columns={
        'Pos': 'Position',
        'Squad': 'Team',
        'MP': 'Games',
        'Gls': 'Goals',
        'Ast': 'Assists',
        'xG+xAG': 'xG + xAG per 90',
        'SCA90': 'SCA per 90'
    }, inplace=True)

    conv = df_output.columns.drop(['Player', 'Position', 'Team'])
    df_output[conv] = df_output[conv].apply(pd.to_numeric, errors='coerce')

    df_output = df_output[df_output['Position'].str.contains("FW|MF")==True]
    df_output = df_output.reset_index(drop=True)

    display(df_output.head(5))
    return df_output

def overall_perf_def(df_playingtime, df_def, df_pass):
    # Filter to necessary columns
    df_playingtime = df_playingtime[['Player','Pos','Squad','MP', 'onGA']]
    df_def = df_def[['Player','Tkl+Int']]
    df_pass = df_pass[['Player','Cmp%','PrgP']]

    df_output = df_def.merge(df_pass, on='Player', how='left')
    df_output = df_output.merge(df_playingtime, on='Player', how='left')

    df_output = df_output[['Player','Pos','Squad','MP', 'onGA', 'Tkl+Int', 'Cmp%', 'PrgP']]
    df_output.rename(columns={
        'Pos': 'Position',
        'Squad': 'Team',
        'MP': 'Games',
        'onGA': 'Goals Conceded',
        'Tkl+Int': 'Tackles + Interceptions',
        'Cmp%': 'Pass Completion %',
        'PrgP': 'Progressive Passes'
    }, inplace=True)

    conv = df_output.columns.drop(['Player', 'Position', 'Team'])
    df_output[conv] = df_output[conv].apply(pd.to_numeric, errors='coerce')

    df_output = df_output[df_output['Position'].str.contains("DF")==True]
    df_output = df_output.reset_index(drop=True)

    display(df_output.head(5))
    return df_output

def shooting(df_shoot, df_pos, df_stat):
    # Filter to necessary columns
    df_stat = df_stat[['Player','Pos','Squad','MP', 'Gls']]
    df_shoot = df_shoot[['Player','np:G-xG', 'SoT']]
    df_pos = df_pos[['Player','Touches']]

    df_output = df_shoot.merge(df_pos, on='Player', how='left')
    df_output = df_output.merge(df_stat, on='Player', how='left')

    df_output = df_output[['Player','Pos','Squad','MP', 'Gls', 'np:G-xG', 'SoT', 'Touches']]
    df_output.rename(columns={
        'Pos': 'Position',
        'Squad': 'Team',
        'MP': 'Games',
        'Gls': 'Goals',
        'np:G-xG': 'Non-Pen xG (+/-)',
        'SoT': 'Shots on Target'
    }, inplace=True)

    conv = df_output.columns.drop(['Player', 'Position', 'Team'])
    df_output[conv] = df_output[conv].apply(pd.to_numeric, errors='coerce')

    display(df_output.head(5))

    return df_output

def prog_types(df_stat, df_pass, df_pos, df_misc):
    # Filter to necessary columns
    df_stat = df_stat[['Player','Pos','Squad','MP']]
    df_pass = df_pass[['Player','PrgP']]
    df_pos = df_pos[['Player','PrgC','Succ']]
    df_misc = df_misc[['Player','Fld']]

    df_output = df_pass.merge(df_pos, on='Player', how='left')
    df_output = df_output.merge(df_misc, on='Player', how='left')
    df_output = df_output.merge(df_stat, on='Player', how='left')

    df_output = df_output[['Player','Pos','Squad','MP','PrgP', 'PrgC', 'Succ', 'Fld']]
    df_output.rename(columns={
        'Pos': 'Position',
        'Squad': 'Team',
        'MP': 'Games',
        'PrgP': 'Progressive Passes',
        'PrgC': 'Progressive Carries',
        'Succ': 'Successful Take-Ons',
        'Fld': 'Fouls Drawn'
    }, inplace=True)

    conv = df_output.columns.drop(['Player', 'Position', 'Team'])
    df_output[conv] = df_output[conv].apply(pd.to_numeric, errors='coerce')

    display(df_output.head(5))

    return df_output

def chance_creation(df_stat, df_gca):
    # Filter to necessary columns
    df_stat = df_stat[['Player','Pos','Squad','MP']]
    df_gca = df_gca[['Player','PassLive','PassDead', 'TO', 'Fld', 'Def']]

    df_output = df_stat.merge(df_gca, on='Player', how='left')

    df_output = df_output[['Player','Pos','Squad','MP','PassLive','PassDead', 'TO', 'Fld', 'Def']]
    df_output.rename(columns={
        'Pos': 'Position',
        'Squad': 'Team',
        'MP': 'Games',
        'PassLive': 'SCA Live Pass',
        'PassDead': 'SCA Set Piece',
        'TO': 'SCA Take-Ons',
        'Fld': 'SCA Foul',
        'Def': 'SCA Defensive'
    }, inplace=True)

    conv = df_output.columns.drop(['Player', 'Position', 'Team'])
    df_output[conv] = df_output[conv].apply(pd.to_numeric, errors='coerce')

    display(df_output.head(5))
    return df_output

def on_ball(df_stat, df_pass, df_shoot, df_pos, df_def):
    # Filter to necessary columns
    df_stat = df_stat[['Player','Pos','Squad','MP']]
    df_pass = df_pass[['Player','Att']]
    df_shoot = df_shoot[['Player','Sh']]
    df_pos = df_pos[['Player','Carries']]
    df_def = df_def[['Player','Tkl+Int']]

    df_output = df_stat.merge(df_pass, on='Player', how='left')
    df_output = df_output.merge(df_shoot, on='Player', how='left')
    df_output = df_output.merge(df_pos, on='Player', how='left')
    df_output = df_output.merge(df_def, on='Player', how='left')

    df_output = df_output[['Player','Pos','Squad','MP', 'Att', 'Sh', 'Carries', 'Tkl+Int']]
    df_output.rename(columns={
        'Pos': 'Position',
        'Squad': 'Team',
        'MP': 'Games',
        'Att': 'Pass Attempts',
        'Sh': 'Shot Attempts',
        'Tkl+Int': 'Tackles + Interceptions'
    }, inplace=True)

    conv = df_output.columns.drop(['Player', 'Position', 'Team'])
    df_output[conv] = df_output[conv].apply(pd.to_numeric, errors='coerce')

    display(df_output.head(5))
    return df_output

def quality_ind(df_stat, df_pass, df_shoot, df_pos, df_def):
    # Filter to necessary columns
    df_stat = df_stat[['Player','Pos','Squad','MP']]
    df_pass = df_pass[['Player','Cmp%']]

    df_shoot = df_shoot[['Player','G/Sh']]
    df_shoot["G/Sh"] = pd.to_numeric(df_shoot["G/Sh"], errors='coerce')
    df_shoot["G/Sh"] = df_shoot["G/Sh"].fillna(0)
    df_shoot["G/Sh"] = df_shoot["G/Sh"].astype('float64')
    df_shoot['Goal to Shot %'] = df_shoot['G/Sh']*100

    df_pos = df_pos[['Player','Carries', 'Mis', 'Dis']]
    df_pos['Carries'] = pd.to_numeric(df_pos['Carries'], errors='coerce')
    df_pos['Carries'] = df_pos['Carries'].fillna(0)

    df_pos['Mis'] = pd.to_numeric(df_pos['Mis'], errors='coerce')
    df_pos['Mis'] = df_pos['Mis'].fillna(0)

    df_pos['Dis'] = pd.to_numeric(df_pos['Dis'], errors='coerce')
    df_pos['Dis'] = df_pos['Dis'].fillna(0)

    df_pos['Carry Success %'] = round(100*df_pos['Carries'] / (df_pos['Carries'] + df_pos['Mis'] + df_pos['Dis']),2)

    df_def = df_def[['Player','Err']]

    df_output = df_stat.merge(df_pass, on='Player', how='left')
    df_output = df_output.merge(df_shoot, on='Player', how='left')
    df_output = df_output.merge(df_pos, on='Player', how='left')
    df_output = df_output.merge(df_def, on='Player', how='left')

    df_output['MP'] = df_output['MP'].astype('int64')

    df_output = df_output[['Player','Pos','Squad','MP', 'Cmp%', 'Goal to Shot %', 'Carry Success %','Err']]
    df_output.rename(columns={
        'Pos': 'Position',
        'Squad': 'Team',
        'MP': 'Games',
        'Cmp%': 'Pass Completion %',
        'Err': 'Defensive Errors',
    }, inplace=True)

    conv = df_output.columns.drop(['Player', 'Position', 'Team'])
    df_output[conv] = df_output[conv].apply(pd.to_numeric, errors='coerce')

    display(df_output.head(5))
    return df_output

def goalkeeping(df_keeper, df_keepadv, df_team_pos):
    # Filter to necessary columns
    df_keeper = df_keeper[['Player','Pos','Squad','MP', 'CS']]
    df_keepadv = df_keepadv[['Player','PSxG+/-','#OPA']]
    df_team_pos = df_team_pos[['Squad', 'Poss']]

    df_output = df_keeper.merge(df_keepadv, on='Player', how='left')
    df_output = df_output.merge(df_team_pos, on='Squad', how='left')

    df_output['CS Rk'] = df_output['CS'].rank(method='min',ascending=False)
    df_output['Team Poss. Rk'] = df_output['Poss'].rank(method='min', ascending=False)
    df_output['PSxG+/- Rk'] = df_output['PSxG+/-'].rank(method='min',ascending=False)
    df_output['Def. Actions Rk'] = df_output['#OPA'].rank(method='min',ascending=False)

    df_output = df_output[['Player','Pos','Squad','MP', 'CS','CS Rk','Poss','Team Poss. Rk', 'PSxG+/-','PSxG+/- Rk','#OPA', 'Def. Actions Rk']]
    df_output.rename(columns={
        'Pos': 'Position',
        'Squad': 'Team',
        'MP': 'Games',
        'CS': 'Clean Sheets',
        'Poss':'Team Possession %',
        'PSxG+/-': 'PS xG (+/-)',
        '#OPA': 'Defensive Actions Outside Box',
    }, inplace=True)

    conv = df_output.columns.drop(['Player', 'Position', 'Team'])
    df_output[conv] = df_output[conv].apply(pd.to_numeric, errors='coerce')

    df_output = df_output.loc[df_output['Position'] == 'GK']
    df_output = df_output.reset_index(drop=True)

    display(df_output.head(5))
    return df_output