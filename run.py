# Project: FbRef Web Scraper
# Author: Abhijot Singh
# Date: 03/01/2024

# File: run.py
# Purpose: File will be run using streamlit to initiate a web-based application for the data scrape.

# Import all official libraries
import pandas as pd
import numpy as np
import streamlit as st
import streamlit_pandas as sp
import importlib
# Import the modules created in the scraper.py file
import scraper
from scraper import *
from constants import *
importlib.reload(scraper)

positions_fbref = pd.read_csv('fb_ref_positions.csv')

# App Headers and subtitles
st.title('FBRef Web Scraper')
st.subheader("Search Europe's Top 5 Leagues by different metrics")
st.write("Data reloads each time the league checkbox is changed. This may take a couple minutes.")

# Checkbox for the different leagues that the user can filter to
league = st.selectbox(
    'Which top 5 league are you looking at?',
    ('Premier League', 'Bundesliga', 'Serie A', 'La Liga', 'Ligue 1', 'Top 5 Europe'))

# Each time the league checkbox is changed, the tables are extracted and stored in cache. If already in cache, they do not re-run.
df_stat, df_gca, df_playingtime, df_def, df_pass, df_shoot, df_pos, df_team_pos, df_misc, df_keeper, df_keepadv = table_extracts(league)

# Selection for the different metrics highlighting the player performance stats
metric = st.selectbox(
    'Which metric do you want data for?',
    ['Overall Performance (Mid / Att)', 'Overall Performance (Def)','Goalkeeping', 'Shooting', 'Progression Type',
     'Chance Creation', 'On Ball Activity', 'Quality Index'])

# Set of cases for each metric
if metric == 'Overall Performance (Mid / Att)':
    df_output = overall_perf_att(df_stat, df_gca)
if metric == 'Overall Performance (Def)':
    df_output = overall_perf_def(df_playingtime, df_def, df_pass)
if metric == 'Goalkeeping':
    df_output = goalkeeping(df_keeper, df_keepadv, df_team_pos)
if metric == 'Shooting':
    df_output = shooting(df_shoot, df_pos, df_stat)
if metric == 'Progression Type':
    df_output = prog_types(df_stat, df_pass, df_pos, df_misc)
if metric == 'Chance Creation':
    df_output = chance_creation(df_stat, df_gca)
if metric == 'On Ball Activity':
    df_output = on_ball(df_stat, df_pass, df_shoot, df_pos, df_def)
if metric == 'Quality Index':
    df_output = quality_ind(df_stat, df_pass, df_shoot, df_pos, df_def)

# Add in Position values derived from: https://docs.google.com/spreadsheets/d/1ksm2fiPBxCBFa0C991ac6ncm2lWKAjCJ-4ganLH_kN0/edit#gid=1745755217
df_output = fuzz_squad(positions_fbref, df_output)

# Specific columns require custom changes to their filtering method
create_data = {
    "Position":'multiselect',
    "Role":'multiselect',
    "Team":'multiselect'
}

# Create filtering widgets on the left of the dataframe for all necessary columns
all_widgets = sp.create_widgets(df_output, create_data)
# Create a filtered form of the dataframe and write this to streamlit
res = sp.filter_df(df_output, all_widgets)
st.dataframe(res)
# Show total number of rows in the filtered metric view prior to the download button
st.write(f'Total Rows: {len(res)}')

# Create Rankings with Spider Ranks here


# Create file name for the unfiltered metric view
file_n = str(metric) + '_' + str(league) + '_' + str(currDay) + str(currMonth) + str(currYear) + '.csv'
# Convert the unfiltered data to necessary download format
df_download = df_output.to_csv(index=False).encode("utf-8")
# Generate a button to download the complete unfiltered metric for the league selected
st.download_button(
    label='Download metric as CSV',
    data=df_download,
    file_name=file_n,
    mime='text/csv'
)

# Repeat process for the filtered version of the metric
file_n_filt = 'Filt' + '_' + str(metric) + '_' + str(league) + '_' + str(currDay) + str(currMonth) + str(currYear) + '.csv'
df_filt_download = res.to_csv(index=False).encode("utf-8")

st.download_button(
    label='Download filtered view as CSV',
    data=df_filt_download,
    file_name=file_n_filt,
    mime='text/csv'
)