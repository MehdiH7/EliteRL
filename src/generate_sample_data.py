import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_eliteserien_data():
    """
    Generate synthetic match data for Eliteserien from 2019-2024.
    """
    # Define teams for each season with promotions and relegations
    teams_by_season = {
        2019: [
            "Bodø/Glimt", "Molde", "Rosenborg", "Odd", "Viking", 
            "Kristiansund", "Brann", "Haugesund", "Strømsgodset", 
            "Vålerenga", "Sarpsborg 08", "Mjøndalen", "Tromsø", 
            "Lillestrøm", "Ranheim", "Stabæk"
        ],
        2020: [
            "Bodø/Glimt", "Molde", "Rosenborg", "Odd", "Viking", 
            "Kristiansund", "Brann", "Haugesund", "Strømsgodset", 
            "Vålerenga", "Sarpsborg 08", "Mjøndalen", "Stabæk", 
            "Sandefjord", "Start", "Aalesund"
        ],
        2021: [
            "Bodø/Glimt", "Molde", "Rosenborg", "Lillestrøm", "Viking", 
            "Kristiansund", "Strømsgodset", "Vålerenga", "Haugesund", 
            "Odd", "Sarpsborg 08", "Tromsø", "Sandefjord", 
            "Mjøndalen", "Stabæk", "Brann"
        ],
        2022: [
            "Molde", "Bodø/Glimt", "Rosenborg", "Lillestrøm", "Viking", 
            "Strømsgodset", "Sarpsborg 08", "Vålerenga", "Haugesund", 
            "Odd", "Kristiansund", "Tromsø", "Sandefjord", 
            "Jerv", "Aalesund", "HamKam"
        ],
        2023: [
            "Bodø/Glimt", "Brann", "Molde", "Tromsø", "Viking", 
            "Rosenborg", "Lillestrøm", "Strømsgodset", "Odd", 
            "KFUM Oslo", "Haugesund", "Sarpsborg 08", "Vålerenga", 
            "HamKam", "Sandefjord", "Stabæk"
        ],
        2024: [
            "Bodø/Glimt", "Brann", "Molde", "Viking", "Rosenborg", 
            "Lillestrøm", "Strømsgodset", "Odd", "Fredrikstad", 
            "Sarpsborg 08", "Haugesund", "KFUM Oslo", "Tromsø", 
            "HamKam", "Kristiansund", "Sandefjord"
        ]
    }
    
    # Assign true strength to each team (will be used to generate realistic match outcomes)
    all_teams = list(set([team for teams in teams_by_season.values() for team in teams]))
    team_strengths = {team: np.random.normal(50, 15) for team in all_teams}
    
    # Some teams should be consistently strong/weak across seasons
    for team in ["Bodø/Glimt", "Molde", "Rosenborg"]:
        team_strengths[team] += 20
    
    # Generate matches for each season
    all_matches = []
    
    for season in range(2019, 2025):
        teams = teams_by_season[season]
        num_teams = len(teams)
        
        # Each team plays against every other team twice (home and away)
        for round_num in range(1, (num_teams-1)*2 + 1):
            # Generate dates for the season (roughly March to November)
            start_date = datetime(season, 3, 1)
            days_offset = (round_num - 1) * 7  # One round per week
            match_date = start_date + timedelta(days=days_offset)
            
            # For each round, generate matches
            matches_in_round = []
            
            # In each round, half of the teams play at home
            np.random.shuffle(teams)
            for i in range(num_teams // 2):
                home_team = teams[i]
                away_team = teams[i + num_teams // 2]
                
                # Calculate match outcome based on team strengths and home advantage
                home_strength = team_strengths[home_team] + 5  # Home advantage
                away_strength = team_strengths[away_team]
                
                # Add some randomness to make it realistic
                home_performance = home_strength + np.random.normal(0, 10)
                away_performance = away_strength + np.random.normal(0, 10)
                
                # Generate goals based on team performances
                home_goals = max(0, int(np.random.poisson(home_performance / 10)))
                away_goals = max(0, int(np.random.poisson(away_performance / 10)))
                
                # Cap goals at a reasonable number
                home_goals = min(home_goals, 7)
                away_goals = min(away_goals, 7)
                
                # Create match record
                match = {
                    'season': season,
                    'date': match_date.strftime('%Y-%m-%d'),
                    'round': round_num,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'home_win': 1 if home_goals > away_goals else 0,
                    'draw': 1 if home_goals == away_goals else 0,
                    'away_win': 1 if home_goals < away_goals else 0
                }
                
                matches_in_round.append(match)
            
            all_matches.extend(matches_in_round)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_matches)
    
    # Ensure each team plays against every other team twice in each season
    for season in range(2019, 2025):
        season_df = df[df['season'] == season]
        teams = teams_by_season[season]
        
        for team in teams:
            home_matches = len(season_df[season_df['home_team'] == team])
            away_matches = len(season_df[season_df['away_team'] == team])
            
            if home_matches != len(teams) - 1 or away_matches != len(teams) - 1:
                print(f"Warning: In season {season}, {team} has {home_matches} home matches and {away_matches} away matches")
    
    return df

def save_data(df, output_path):
    """
    Save the generated data to CSV files.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save all data
    df.to_csv(output_path, index=False)
    
    # Also save data split by seasons
    for season in range(2019, 2025):
        season_df = df[df['season'] == season]
        season_path = output_path.replace('.csv', f'_{season}.csv')
        season_df.to_csv(season_path, index=False)
    
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Generate data
    match_data = generate_eliteserien_data()
    
    # Save data
    save_data(match_data, 'data/raw/eliteserien_matches.csv') 