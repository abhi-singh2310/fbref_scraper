# Project: FbRef Web Scraper
# Author: Abhijot Singh
# Date: 03/01/2024

# File: constants.py
# Purpose: Contains league names and stat headers to use for url links

from datetime import datetime

# Current Date
currDay = datetime.now().day
currMonth = datetime.now().month
currYear = datetime.now().year

# League Strings for URL
PREM = 'Premier-League-Stats'
BUND = 'Bundesliga-Stats'
LALIGA = 'La-Liga-Stats'
SERIEA = 'Serie-A-Stats'
LIGUE_1 = 'Ligue-1-Stats'

# Top 5 leagues combined
ALL = 'Big-5-European-Leagues-Stats'

