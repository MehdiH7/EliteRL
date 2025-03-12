import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import modules
from src.process_real_data import prepare_real_data_for_rl, load_all_seasons
from src.data_processing import calculate_season_standings
from src.rl_model import DuelingBanditRL, BradleyTerryModel
from src.traditional_models import LogisticRegressionModel, RandomForestModel
from src.evaluation import (
    calculate_prediction_accuracy,
    compare_rankings,
    plot_ranking_comparison,
    plot_prediction_accuracy,
    plot_team_strength_evolution,
    save_rankings_to_csv,
    compare_rank_distances
)

def main():
    # Create output directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Process real data
    print("Processing real Eliteserien data...")
    train_comparisons, test_comparisons, preference_matrix, team_to_idx, active_teams = prepare_real_data_for_rl(
        data_dir='data/raw',
        output_dir='data/processed',
        training_seasons=(2019, 2020, 2021, 2022, 2023),
        test_season=2024
    )
    
    # Load all match data for standings calculation
    all_match_data = load_all_seasons('data/raw')
    
    # Create inverse mapping
    idx_to_team = {idx: team for team, idx in team_to_idx.items()}
    
    # Step 2: Calculate actual 2024 standings
    print("\nCalculating actual 2024 standings...")
    actual_standings_2024 = calculate_season_standings(all_match_data, 2024)
    actual_ranking_2024 = actual_standings_2024.index.tolist()
    
    # Print actual 2024 standings
    print("\nActual 2024 Standings:")
    print(actual_standings_2024[['played', 'wins', 'draws', 'losses', 'points', 'goal_diff', 'rank']])
    
    # Step 3: Train RL model
    print("\nTraining Dueling Bandit RL model...")
    rl_model = DuelingBanditRL(
        n_teams=len(active_teams),
        alpha=0.05,  # Lower initial learning rate for stability
        exploration_factor=2.0,  # Increased exploration
        use_features=True,
        feature_learning_rate=0.02,  # Faster feature learning
        decay_rate=0.9995,  # Slower decay
        min_alpha=0.005,  # Lower minimum learning rate
        use_thompson_sampling=True,  # Use Thompson sampling for better exploration
        use_double_learning=True  # Use double learning for stability
    )
    rl_rankings_indices = rl_model.train_with_real_matches(train_comparisons, team_to_idx)
    
    # Convert RL rankings from indices to team names
    rl_rankings = [idx_to_team[idx] for idx in rl_rankings_indices]
    
    # Step 4: Train Bradley-Terry model
    print("\nTraining Bradley-Terry model...")
    bt_model = BradleyTerryModel(n_teams=len(active_teams), learning_rate=0.01, n_iterations=1000)
    bt_model.train(train_comparisons, team_to_idx)
    
    # Get Bradley-Terry rankings
    bt_rankings = [idx_to_team[idx] for idx in bt_model.get_rankings()]
    
    # Step 5: Train traditional models with enhanced features
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegressionModel(C=0.5)  # Adjust regularization
    lr_model.train(train_comparisons, team_to_idx)
    
    print("\nTraining Random Forest model...")
    rf_model = RandomForestModel(n_estimators=200, max_depth=10)  # More trees, limited depth
    rf_model.train(train_comparisons, team_to_idx)
    
    # Get traditional model rankings
    lr_rankings = lr_model.get_rankings([idx_to_team[i] for i in range(len(active_teams))])
    rf_rankings = rf_model.get_rankings([idx_to_team[i] for i in range(len(active_teams))])
    
    # Step 6: Evaluate models
    print("\nEvaluating models...")
    
    # Calculate prediction accuracy
    rl_accuracy = rl_model.evaluate(test_comparisons, team_to_idx)
    bt_accuracy = bt_model.evaluate(test_comparisons, team_to_idx)
    lr_accuracy = lr_model.evaluate(test_comparisons, team_to_idx)
    rf_accuracy = rf_model.evaluate(test_comparisons, team_to_idx)
    
    accuracy_dict = {
        'RL': rl_accuracy,
        'Bradley-Terry': bt_accuracy,
        'Logistic Regression': lr_accuracy,
        'Random Forest': rf_accuracy
    }
    
    print("\nPrediction Accuracy:")
    for model, accuracy in accuracy_dict.items():
        print(f"{model}: {accuracy:.4f}")
    
    # Compare rankings
    rankings_dict = {
        'RL': rl_rankings,
        'Bradley-Terry': bt_rankings,
        'Logistic Regression': lr_rankings,
        'Random Forest': rf_rankings
    }
    
    # Print model rankings
    print("\nModel Rankings:")
    for model, ranking in rankings_dict.items():
        print(f"\n{model} Ranking:")
        for i, team in enumerate(ranking[:10]):  # Show top 10
            print(f"{i+1}. {team}")
    
    # Calculate ranking correlations
    ranking_correlations = compare_rankings(rankings_dict, actual_ranking_2024, method='kendall')
    
    print("\nRanking Correlations (Kendall's Tau):")
    print(ranking_correlations)
    
    # Calculate rank distances
    rank_distances = compare_rank_distances(rankings_dict, actual_ranking_2024)
    
    print("\nRank Distances (Average absolute difference in ranks):")
    print(rank_distances)
    
    # After evaluating all models, print feature importance for RL model
    print("\nRL Model Feature Importance:")
    feature_importance = rl_model.get_feature_importance()
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.4f}")
    
    # Step 7: Generate plots
    print("\nGenerating plots...")
    
    # Plot ranking comparison
    ranking_plot = plot_ranking_comparison(rankings_dict, actual_ranking_2024, title='Ranking Comparison with 2024 Actual Standings')
    ranking_plot.savefig('results/ranking_comparison.png')
    
    # Plot prediction accuracy
    accuracy_plot = plot_prediction_accuracy(accuracy_dict, title='Match Prediction Accuracy')
    accuracy_plot.savefig('results/prediction_accuracy.png')
    
    # Plot team strength evolution
    evolution_plot = plot_team_strength_evolution(rl_model, team_to_idx, idx_to_team, title='Team Strength Evolution during RL Training')
    evolution_plot.savefig('results/team_strength_evolution.png')
    
    # Save rankings to CSV
    save_rankings_to_csv(rankings_dict, actual_ranking_2024, 'results/rankings.csv')
    
    print("\nResults saved to 'results' directory.")

if __name__ == "__main__":
    main() 