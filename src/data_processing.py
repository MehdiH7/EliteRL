import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """
    Load match data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing match data.
        
    Returns:
        pd.DataFrame: DataFrame containing match data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return pd.read_csv(file_path)

def filter_seasons(df, seasons):
    """
    Filter data to include only specified seasons.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data.
        seasons (list): List of seasons to include.
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return df[df['season'].isin(seasons)]

def create_pairwise_comparisons(df):
    """
    Convert match results to pairwise comparisons with enhanced features.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data.
        
    Returns:
        pd.DataFrame: DataFrame with pairwise comparisons and additional features.
    """
    comparisons = []
    
    # Sort by date to calculate form
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
        df = df.sort_values('date')
    
    # Track team form (last 5 matches)
    team_form = {}
    team_home_form = {}
    team_away_form = {}
    
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_goals = row['home_goals']
        away_goals = row['away_goals']
        season = row['season']
        
        # Initialize form if not exists
        if home_team not in team_form:
            team_form[home_team] = []
            team_home_form[home_team] = []
            team_away_form[home_team] = []
        
        if away_team not in team_form:
            team_form[away_team] = []
            team_home_form[away_team] = []
            team_away_form[away_team] = []
        
        # Calculate current form (last 5 matches)
        home_form = sum(team_form[home_team][-5:]) / max(len(team_form[home_team][-5:]), 1)
        away_form = sum(team_form[away_team][-5:]) / max(len(team_form[away_team][-5:]), 1)
        
        # Calculate home/away specific form
        home_home_form = sum(team_home_form[home_team][-3:]) / max(len(team_home_form[home_team][-3:]), 1)
        away_away_form = sum(team_away_form[away_team][-3:]) / max(len(team_away_form[away_team][-3:]), 1)
        
        # Determine match outcome
        if home_goals > away_goals:
            winner = home_team
            loser = away_team
            is_draw = False
            # Update form
            team_form[home_team].append(3)  # Win = 3 points
            team_home_form[home_team].append(3)
            team_form[away_team].append(0)  # Loss = 0 points
            team_away_form[away_team].append(0)
        elif away_goals > home_goals:
            winner = away_team
            loser = home_team
            is_draw = False
            # Update form
            team_form[home_team].append(0)  # Loss = 0 points
            team_home_form[home_team].append(0)
            team_form[away_team].append(3)  # Win = 3 points
            team_away_form[away_team].append(3)
        else:
            # In case of a draw, we'll create two comparisons with draw flag
            winner = home_team
            loser = away_team
            is_draw = True
            # Update form
            team_form[home_team].append(1)  # Draw = 1 point
            team_home_form[home_team].append(1)
            team_form[away_team].append(1)  # Draw = 1 point
            team_away_form[away_team].append(1)
        
        # Create comparison record with enhanced features
        comparison = {
            'season': season,
            'winner': winner,
            'loser': loser,
            'goal_diff': abs(home_goals - away_goals),
            'is_draw': is_draw,
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_form': home_form,
            'away_form': away_form,
            'home_home_form': home_home_form,
            'away_away_form': away_away_form,
            'total_goals': home_goals + away_goals
        }
        
        comparisons.append(comparison)
        
        # For draws, add the reverse comparison as well
        if is_draw:
            reverse_comparison = comparison.copy()
            reverse_comparison['winner'] = away_team
            reverse_comparison['loser'] = home_team
            comparisons.append(reverse_comparison)
    
    return pd.DataFrame(comparisons)

def create_preference_matrix(df, teams):
    """
    Create a preference matrix from pairwise comparisons.
    
    Args:
        df (pd.DataFrame): DataFrame with pairwise comparisons.
        teams (list): List of all teams.
        
    Returns:
        np.ndarray: Preference matrix where P[i,j] is the probability that team i beats team j.
    """
    n_teams = len(teams)
    team_to_idx = {team: i for i, team in enumerate(teams)}
    
    # Initialize preference matrix with 0.5 (equal probability)
    P = np.ones((n_teams, n_teams)) * 0.5
    
    # Set diagonal to 0 (a team doesn't play against itself)
    np.fill_diagonal(P, 0)
    
    # Count wins and total matches
    wins = np.zeros((n_teams, n_teams))
    matches = np.zeros((n_teams, n_teams))
    
    for _, row in df.iterrows():
        winner = row['winner']
        loser = row['loser']
        
        if winner in team_to_idx and loser in team_to_idx:
            i = team_to_idx[winner]
            j = team_to_idx[loser]
            
            wins[i, j] += 1
            matches[i, j] += 1
            matches[j, i] += 1  # Count the match for both teams
    
    # Calculate win probabilities
    for i in range(n_teams):
        for j in range(n_teams):
            if matches[i, j] > 0:
                P[i, j] = wins[i, j] / matches[i, j]
    
    return P, team_to_idx

def get_active_teams(df, min_seasons=2):
    """
    Get list of teams that have played in at least min_seasons seasons.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data.
        min_seasons (int): Minimum number of seasons a team must have played.
        
    Returns:
        list: List of active teams.
    """
    # Get all unique teams
    home_teams = set(df['home_team'].unique())
    away_teams = set(df['away_team'].unique())
    all_teams = home_teams.union(away_teams)
    
    # Count seasons for each team
    team_seasons = {}
    for team in all_teams:
        seasons_home = set(df[df['home_team'] == team]['season'])
        seasons_away = set(df[df['away_team'] == team]['season'])
        team_seasons[team] = len(seasons_home.union(seasons_away))
    
    # Filter teams that have played in at least min_seasons seasons
    active_teams = [team for team, count in team_seasons.items() if count >= min_seasons]
    
    return active_teams

def prepare_data_for_rl(raw_data_path, processed_data_path, training_seasons=(2019, 2020, 2021, 2022, 2023), test_season=2024):
    """
    Prepare data for RL model.
    
    Args:
        raw_data_path (str): Path to raw data file.
        processed_data_path (str): Path to save processed data.
        training_seasons (tuple): Seasons to use for training.
        test_season (int): Season to use for testing.
        
    Returns:
        tuple: (training_comparisons, test_comparisons, preference_matrix, team_to_idx, active_teams)
    """
    # Load data
    df = load_data(raw_data_path)
    
    # Filter training and test data
    train_df = filter_seasons(df, training_seasons)
    test_df = filter_seasons(df, [test_season])
    
    # Get active teams (teams that have played in at least 2 seasons)
    active_teams = get_active_teams(df, min_seasons=2)
    
    # Create pairwise comparisons
    train_comparisons = create_pairwise_comparisons(train_df)
    test_comparisons = create_pairwise_comparisons(test_df)
    
    # Create preference matrix
    preference_matrix, team_to_idx = create_preference_matrix(train_comparisons, active_teams)
    
    # Save processed data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    # Save training comparisons
    train_comparisons.to_csv(processed_data_path.replace('.csv', '_train_comparisons.csv'), index=False)
    
    # Save test comparisons
    test_comparisons.to_csv(processed_data_path.replace('.csv', '_test_comparisons.csv'), index=False)
    
    # Save preference matrix
    np.save(processed_data_path.replace('.csv', '_preference_matrix.npy'), preference_matrix)
    
    # Save team mapping
    pd.DataFrame({
        'team': list(team_to_idx.keys()),
        'index': list(team_to_idx.values())
    }).to_csv(processed_data_path.replace('.csv', '_team_mapping.csv'), index=False)
    
    return train_comparisons, test_comparisons, preference_matrix, team_to_idx, active_teams

def calculate_season_standings(df, season):
    """
    Calculate the final standings for a given season.
    
    Args:
        df (pd.DataFrame): DataFrame containing match data.
        season (int): Season to calculate standings for.
        
    Returns:
        pd.DataFrame: DataFrame with team standings.
    """
    # Filter data for the specified season
    season_df = df[df['season'] == season]
    
    # Get all teams in the season
    home_teams = set(season_df['home_team'].unique())
    away_teams = set(season_df['away_team'].unique())
    all_teams = list(home_teams.union(away_teams))
    
    # Initialize standings dictionary
    standings = {team: {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0, 
                        'goals_for': 0, 'goals_against': 0, 'points': 0} 
                for team in all_teams}
    
    # Calculate statistics for each match
    for _, match in season_df.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        home_goals = match['home_goals']
        away_goals = match['away_goals']
        
        # Update home team stats
        standings[home_team]['played'] += 1
        standings[home_team]['goals_for'] += home_goals
        standings[home_team]['goals_against'] += away_goals
        
        # Update away team stats
        standings[away_team]['played'] += 1
        standings[away_team]['goals_for'] += away_goals
        standings[away_team]['goals_against'] += home_goals
        
        # Update results based on match outcome
        if home_goals > away_goals:
            standings[home_team]['wins'] += 1
            standings[home_team]['points'] += 3
            standings[away_team]['losses'] += 1
        elif home_goals < away_goals:
            standings[away_team]['wins'] += 1
            standings[away_team]['points'] += 3
            standings[home_team]['losses'] += 1
        else:
            standings[home_team]['draws'] += 1
            standings[home_team]['points'] += 1
            standings[away_team]['draws'] += 1
            standings[away_team]['points'] += 1
    
    # Convert to DataFrame
    standings_df = pd.DataFrame.from_dict(standings, orient='index')
    
    # Add goal difference
    standings_df['goal_diff'] = standings_df['goals_for'] - standings_df['goals_against']
    
    # Sort by points, then goal difference
    standings_df = standings_df.sort_values(['points', 'goal_diff'], ascending=False)
    
    # Add rank
    standings_df['rank'] = range(1, len(standings_df) + 1)
    
    return standings_df 