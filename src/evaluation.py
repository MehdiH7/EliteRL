import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau, spearmanr

def calculate_ranking_correlation(ranking1, ranking2, method='kendall'):
    """
    Calculate correlation between two rankings.
    
    Args:
        ranking1 (list): First ranking.
        ranking2 (list): Second ranking.
        method (str): Correlation method ('kendall' or 'spearman').
        
    Returns:
        float: Correlation coefficient.
    """
    # Convert rankings to ranks
    rank1 = {team: i+1 for i, team in enumerate(ranking1)}
    rank2 = {team: i+1 for i, team in enumerate(ranking2)}
    
    # Get common teams
    common_teams = set(ranking1).intersection(set(ranking2))
    
    # Extract ranks for common teams
    ranks1 = [rank1[team] for team in common_teams]
    ranks2 = [rank2[team] for team in common_teams]
    
    # Calculate correlation
    if method == 'kendall':
        corr, _ = kendalltau(ranks1, ranks2)
    elif method == 'spearman':
        corr, _ = spearmanr(ranks1, ranks2)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    return corr

def calculate_prediction_accuracy(model, test_comparisons, team_to_idx):
    """
    Calculate prediction accuracy on test data.
    
    Args:
        model: Trained model with evaluate method.
        test_comparisons (pd.DataFrame): DataFrame with test pairwise comparisons.
        team_to_idx (dict): Mapping from team names to indices.
        
    Returns:
        float: Prediction accuracy.
    """
    return model.evaluate(test_comparisons, team_to_idx)

def compare_rankings(rankings_dict, reference_ranking, method='kendall'):
    """
    Compare multiple rankings against a reference ranking.
    
    Args:
        rankings_dict (dict): Dictionary of rankings {model_name: ranking}.
        reference_ranking (list): Reference ranking to compare against.
        method (str): Correlation method ('kendall' or 'spearman').
        
    Returns:
        pd.DataFrame: DataFrame with correlation results.
    """
    results = []
    
    for model_name, ranking in rankings_dict.items():
        corr = calculate_ranking_correlation(ranking, reference_ranking, method)
        results.append({
            'model': model_name,
            'correlation': corr
        })
    
    return pd.DataFrame(results)

def plot_ranking_comparison(rankings_dict, reference_ranking, title='Ranking Comparison'):
    """
    Plot comparison of rankings.
    
    Args:
        rankings_dict (dict): Dictionary of rankings {model_name: ranking}.
        reference_ranking (list): Reference ranking to compare against.
        title (str): Plot title.
    """
    # Get common teams across all rankings
    common_teams = set(reference_ranking)
    for ranking in rankings_dict.values():
        common_teams = common_teams.intersection(set(ranking))
    common_teams = list(common_teams)
    
    # Check if there are common teams
    if not common_teams:
        print("Warning: No common teams found across all rankings. Cannot create comparison plot.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No common teams found across all rankings", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.title(title)
        return plt
    
    # Create DataFrame for plotting
    df = pd.DataFrame()
    
    # Add reference ranking
    ref_ranks = {team: i+1 for i, team in enumerate(reference_ranking) if team in common_teams}
    df['Reference'] = [ref_ranks.get(team, np.nan) for team in common_teams]
    
    # Add model rankings
    for model_name, ranking in rankings_dict.items():
        model_ranks = {team: i+1 for i, team in enumerate(ranking) if team in common_teams}
        df[model_name] = [model_ranks.get(team, np.nan) for team in common_teams]
    
    # Set index to team names
    df.index = common_teams
    
    # Sort by reference ranking
    df = df.sort_values('Reference')
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap='YlGnBu', cbar=False)
    plt.title(title)
    plt.ylabel('Teams')
    plt.xlabel('Models')
    plt.tight_layout()
    
    return plt

def plot_prediction_accuracy(accuracy_dict, title='Prediction Accuracy'):
    """
    Plot prediction accuracy for different models.
    
    Args:
        accuracy_dict (dict): Dictionary of accuracies {model_name: accuracy}.
        title (str): Plot title.
    """
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'model': list(accuracy_dict.keys()),
        'accuracy': list(accuracy_dict.values())
    })
    
    # Sort by accuracy
    df = df.sort_values('accuracy', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='accuracy', data=df)
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt

def plot_team_strength_evolution(rl_model, team_to_idx, idx_to_team, title='Team Strength Evolution'):
    """
    Plot evolution of team strengths during RL training.
    
    Args:
        rl_model: Trained RL model with value_history attribute.
        team_to_idx (dict): Mapping from team names to indices.
        idx_to_team (dict): Mapping from indices to team names.
        title (str): Plot title.
    """
    # Get value history
    value_history = np.array(rl_model.value_history)
    
    # Select a subset of teams to plot (top and bottom teams)
    final_values = value_history[-1]
    top_indices = np.argsort(-final_values)[:5]  # Top 5 teams
    bottom_indices = np.argsort(final_values)[:5]  # Bottom 5 teams
    selected_indices = np.concatenate([top_indices, bottom_indices])
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    for idx in selected_indices:
        team_name = idx_to_team[idx]
        plt.plot(value_history[:, idx], label=team_name)
    
    plt.title(title)
    plt.xlabel('Training Iterations')
    plt.ylabel('Team Strength')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    return plt

def save_rankings_to_csv(rankings_dict, reference_ranking, output_path):
    """
    Save rankings to CSV file.
    
    Args:
        rankings_dict (dict): Dictionary of rankings {model_name: ranking}.
        reference_ranking (list): Reference ranking to compare against.
        output_path (str): Path to save CSV file.
    """
    # Create DataFrame
    df = pd.DataFrame()
    
    # Add reference ranking
    df['Reference'] = reference_ranking
    
    # Find common teams across all rankings
    common_teams = set(reference_ranking)
    for ranking in rankings_dict.values():
        common_teams = common_teams.intersection(set(ranking))
    
    # Create a mapping of team to rank for each model
    rank_mappings = {}
    for model_name, ranking in rankings_dict.items():
        rank_mappings[model_name] = {team: i+1 for i, team in enumerate(ranking)}
    
    # Create a DataFrame with ranks for common teams
    result_df = pd.DataFrame(index=reference_ranking)
    result_df['Reference'] = range(1, len(reference_ranking) + 1)
    
    for model_name, rank_map in rank_mappings.items():
        result_df[model_name] = [rank_map.get(team, np.nan) for team in reference_ranking]
    
    # Save to CSV
    result_df.to_csv(output_path)
    
    print(f"Rankings saved to {output_path}")

def calculate_rank_distance(ranking1, ranking2):
    """
    Calculate average absolute difference in ranks between two rankings.
    
    Args:
        ranking1 (list): First ranking.
        ranking2 (list): Second ranking.
        
    Returns:
        float: Average rank distance.
    """
    # Convert rankings to ranks
    rank1 = {team: i+1 for i, team in enumerate(ranking1)}
    rank2 = {team: i+1 for i, team in enumerate(ranking2)}
    
    # Get common teams
    common_teams = set(ranking1).intersection(set(ranking2))
    
    # Calculate absolute differences
    differences = [abs(rank1[team] - rank2[team]) for team in common_teams]
    
    # Return average
    return sum(differences) / len(differences) if differences else 0

def compare_rank_distances(rankings_dict, reference_ranking):
    """
    Compare rank distances between multiple rankings and a reference ranking.
    
    Args:
        rankings_dict (dict): Dictionary of rankings {model_name: ranking}.
        reference_ranking (list): Reference ranking to compare against.
        
    Returns:
        pd.DataFrame: DataFrame with rank distance results.
    """
    results = []
    
    for model_name, ranking in rankings_dict.items():
        distance = calculate_rank_distance(ranking, reference_ranking)
        results.append({
            'model': model_name,
            'rank_distance': distance
        })
    
    return pd.DataFrame(results)