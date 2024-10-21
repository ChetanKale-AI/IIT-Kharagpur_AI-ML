#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import sys

# Function to convert wage string to numeric value
def convert_wage_to_numeric(wage_str):
    # Remove the euro symbol and 'K' suffix
    wage_str = re.sub('[€K]', '', wage_str)
    # Convert to numeric and multiply by 1000
    if wage_str.isdigit():
        return int(wage_str) * 1000
    return 0  # Default to 0 if format is unexpected

# Function to calculate mean overall score of players from a country
def mean_overall_country(df, country):
    # Filter players from the specified country
    country_players = df[df['Nationality'] == country]
    # Calculate and return the mean of the 'Overall' column
    return country_players['Overall'].mean()

# Function to calculate mean wage of players from a club
def mean_wage_club(df, club):
    # Create a copy of the club players slice to avoid SettingWithCopyWarning
    club_players = df[df['Club'] == club].copy()
    # Apply the conversion function to each wage value and assign to a new column
    club_players['Numeric Wage'] = club_players['Wage'].apply(convert_wage_to_numeric)
    # Return the mean of the 'Numeric Wage' column
    return club_players['Numeric Wage'].mean()

# Function to find common players between a club and a country
def common_players(df, club, country):
    # Filter players from the specified country and club
    club_players = df[df['Club'] == club]
    country_players = df[df['Nationality'] == country]
    # Find common players based on their 'Name' column
    common_players = club_players[club_players['Name'].isin(country_players['Name'])]
    # Return the list of player names
    return common_players['Name'].tolist()

# Main function to load data, perform tests, and return formatted results
def main(df, country_name, club_name):

    # Finding common players
    common_players_list = common_players(df, club_name, country_name)

    # Calculating mean overall score of country
    mean_country_overall = float(mean_overall_country(df, country_name))

    # Calculating mean overall wage of club
    mean_club_wage = float(mean_wage_club(df, club_name))

    # Return results as a list with formatted values
    print([", ".join(common_players_list), round(mean_country_overall, 2), round(mean_club_wage, 2)])

if __name__ == "__main__":
    df = pd.read_csv('data.csv').head(100)
    result = main(df, sys.argv[1], sys.argv[2])

