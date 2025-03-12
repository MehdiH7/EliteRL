import pandas as pd
import numpy as np
import os
import glob
from src.data_processing import create_pairwise_comparisons, create_preference_matrix, get_active_teams

def load_season_data(file_path):
    """
    Load and process a single season's data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing match data.
        
    Returns:
        pd.DataFrame: Processed DataFrame containing match data.
    """
    # Extract season from filename
    season = int(file_path.split('_')[-1].split('.')[0])
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Process data
    matches = []
    
    for _, row in df.iterrows():
        # Extract teams
        home_team = row['Hjemmelag']
        away_team = row['Bortelag']
        
        # Extract result
        result = row['Resultat']
        home_goals, away_goals = map(int, result.split(' - '))
        
        # Create match record
        match = {
            'season': season,
            'round': row['Runde'],
            'date': row['Dato'],
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'home_win': 1 if home_goals > away_goals else 0,
            'draw': 1 if home_goals == away_goals else 0,
            'away_win': 1 if home_goals < away_goals else 0
        }
        
        matches.append(match)
    
    return pd.DataFrame(matches)

def load_all_seasons(data_dir='data/raw', pattern='eliteserien_*.csv'):
    """
    Load all seasons' data from CSV files.
    
    Args:
        data_dir (str): Directory containing the CSV files.
        pattern (str): Pattern to match CSV files.
        
    Returns:
        pd.DataFrame: Combined DataFrame containing all match data.
    """
    # Get all CSV files
    file_paths = glob.glob(os.path.join(data_dir, pattern))
    
    # Load and combine data
    all_data = []
    
    for file_path in file_paths:
        season_data = load_season_data(file_path)
        all_data.append(season_data)
    
    return pd.concat(all_data, ignore_index=True)

def prepare_real_data_for_rl(data_dir='data/raw', output_dir='data/processed', 
                           training_seasons=(2019, 2020, 2021, 2022, 2023), test_season=2024):
    """
    Prepare real data for RL model.
    
    Args:
        data_dir (str): Directory containing the raw data files.
        output_dir (str): Directory to save processed data.
        training_seasons (tuple): Seasons to use for training.
        test_season (int): Season to use for testing.
        
    Returns:
        tuple: (training_comparisons, test_comparisons, preference_matrix, team_to_idx, active_teams)
    """
    # Load all data
    all_data = load_all_seasons(data_dir)
    
    # Save combined data
    os.makedirs(output_dir, exist_ok=True)
    all_data.to_csv(os.path.join(output_dir, 'eliteserien_all_matches.csv'), index=False)
    
    # Filter training and test data
    train_df = all_data[all_data['season'].isin(training_seasons)]
    test_df = all_data[all_data['season'] == test_season]
    
    # Get active teams (teams that have played in at least 2 seasons)
    active_teams = get_active_teams(all_data, min_seasons=2)
    
    # Create pairwise comparisons
    train_comparisons = create_pairwise_comparisons(train_df)
    test_comparisons = create_pairwise_comparisons(test_df)
    
    # Create preference matrix
    preference_matrix, team_to_idx = create_preference_matrix(train_comparisons, active_teams)
    
    # Save processed data
    train_comparisons.to_csv(os.path.join(output_dir, 'eliteserien_train_comparisons.csv'), index=False)
    test_comparisons.to_csv(os.path.join(output_dir, 'eliteserien_test_comparisons.csv'), index=False)
    np.save(os.path.join(output_dir, 'eliteserien_preference_matrix.npy'), preference_matrix)
    
    # Save team mapping
    pd.DataFrame({
        'team': list(team_to_idx.keys()),
        'index': list(team_to_idx.values())
    }).to_csv(os.path.join(output_dir, 'eliteserien_team_mapping.csv'), index=False)
    
    return train_comparisons, test_comparisons, preference_matrix, team_to_idx, active_teams

if __name__ == "__main__":
    # Process real data
    train_comparisons, test_comparisons, preference_matrix, team_to_idx, active_teams = prepare_real_data_for_rl()
    
    print(f"Processed {len(train_comparisons)} training comparisons and {len(test_comparisons)} test comparisons")
    print(f"Found {len(active_teams)} active teams across seasons") 