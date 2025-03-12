import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class LogisticRegressionModel:
    """
    Logistic Regression model for predicting match outcomes.
    """
    
    def __init__(self, C=1.0):
        """
        Initialize the Logistic Regression model.
        
        Args:
            C (float): Regularization parameter.
        """
        self.model = LogisticRegression(C=C, max_iter=1000)
        self.team_encoder = None
        self.feature_scaler = StandardScaler()
        self.team_to_idx = None
    
    def _prepare_data(self, comparisons, team_to_idx):
        """
        Prepare data for training/testing with enhanced features.
        
        Args:
            comparisons (pd.DataFrame): DataFrame with pairwise comparisons.
            team_to_idx (dict): Mapping from team names to indices.
            
        Returns:
            tuple: (X, y) - Features and labels.
        """
        X_teams = []
        X_features = []
        y = []
        
        for _, row in comparisons.iterrows():
            winner = row['winner']
            loser = row['loser']
            
            if winner in team_to_idx and loser in team_to_idx:
                # Skip draws for direct prediction (they're handled differently)
                if 'is_draw' in row and row['is_draw']:
                    continue
                
                # Create team identity features
                X_teams.append([winner, loser])
                y.append(1)  # Winner beats loser
                
                X_teams.append([loser, winner])
                y.append(0)  # Loser doesn't beat winner
                
                # Extract additional features if available
                features1 = []
                features2 = []
                
                # Add home advantage
                if 'home_team' in row and 'away_team' in row:
                    is_home1 = 1 if row['home_team'] == winner else 0
                    is_home2 = 1 if row['home_team'] == loser else 0
                    features1.append(is_home1)
                    features2.append(is_home2)
                else:
                    # Default values if home/away info not available
                    features1.append(0.5)  # Neutral
                    features2.append(0.5)  # Neutral
                
                # Add form
                if 'home_form' in row and 'away_form' in row:
                    if row['home_team'] == winner:
                        form1 = row['home_form']
                        form2 = row['away_form']
                    else:
                        form1 = row['away_form']
                        form2 = row['home_form']
                    features1.append(form1)
                    features2.append(form2)
                else:
                    # Default values if form info not available
                    features1.append(0.5)  # Neutral
                    features2.append(0.5)  # Neutral
                
                # Add home/away specific form
                if 'home_home_form' in row and 'away_away_form' in row and 'home_team' in row:
                    if row['home_team'] == winner:
                        home_form1 = row['home_home_form']
                        away_form2 = row['away_away_form']
                    else:
                        away_form1 = row['away_away_form']
                        home_form2 = row['home_home_form']
                    features1.append(home_form1 if row['home_team'] == winner else away_form1)
                    features2.append(away_form2 if row['home_team'] == winner else home_form2)
                else:
                    # Default values if home/away form info not available
                    features1.append(0.5)  # Neutral
                    features2.append(0.5)  # Neutral
                
                # Add goal-related features
                if 'goal_diff' in row:
                    features1.append(row['goal_diff'])
                    features2.append(row['goal_diff'])
                else:
                    # Default value if goal diff not available
                    features1.append(1.0)  # Small difference
                    features2.append(1.0)  # Small difference
                
                if 'total_goals' in row:
                    features1.append(row['total_goals'])
                    features2.append(row['total_goals'])
                else:
                    # Default value if total goals not available
                    features1.append(2.0)  # Average number of goals
                    features2.append(2.0)  # Average number of goals
                
                # Add features for both comparisons
                X_features.append(features1)
                X_features.append(features2)
        
        # Convert to DataFrame
        X_teams_df = pd.DataFrame(X_teams, columns=['team1', 'team2'])
        
        # One-hot encode teams
        if self.team_encoder is None:
            # For training
            self.team_encoder = OneHotEncoder(sparse_output=False)
            team_encoded = self.team_encoder.fit_transform(X_teams_df)
        else:
            # For testing
            team_encoded = self.team_encoder.transform(X_teams_df)
        
        # Combine with additional features if available
        if X_features and len(X_features[0]) > 0:
            X_features = np.array(X_features)
            
            # Scale features
            if self.feature_scaler is not None:
                if len(X_features) == len(team_encoded):
                    if hasattr(self.model, 'coef_'):  # Model is already trained
                        X_features = self.feature_scaler.transform(X_features)
                    else:  # Model is not trained yet
                        X_features = self.feature_scaler.fit_transform(X_features)
                    
                    # Combine team encoding with additional features
                    X_encoded = np.hstack([team_encoded, X_features])
                else:
                    X_encoded = team_encoded
            else:
                X_encoded = np.hstack([team_encoded, X_features])
        else:
            X_encoded = team_encoded
        
        return X_encoded, np.array(y)
    
    def train(self, comparisons, team_to_idx):
        """
        Train the model.
        
        Args:
            comparisons (pd.DataFrame): DataFrame with pairwise comparisons.
            team_to_idx (dict): Mapping from team names to indices.
            
        Returns:
            self: Trained model.
        """
        self.team_to_idx = team_to_idx
        
        X, y = self._prepare_data(comparisons, team_to_idx)
        
        if len(X) > 0:
            self.model.fit(X, y)
        
        return self
    
    def evaluate(self, test_comparisons, team_to_idx):
        """
        Evaluate model on test data.
        
        Args:
            test_comparisons (pd.DataFrame): DataFrame with test pairwise comparisons.
            team_to_idx (dict): Mapping from team names to indices.
            
        Returns:
            float: Accuracy of predictions.
        """
        # Skip draws for evaluation
        non_draw_comparisons = test_comparisons
        if 'is_draw' in test_comparisons.columns:
            non_draw_comparisons = test_comparisons[~test_comparisons['is_draw']]
        
        X, y = self._prepare_data(non_draw_comparisons, team_to_idx)
        
        if len(X) > 0:
            return self.model.score(X, y)
        else:
            return 0.0
    
    def predict_match(self, team1, team2, is_team1_home=True):
        """
        Predict the outcome of a match between two teams.
        
        Args:
            team1 (str): First team.
            team2 (str): Second team.
            is_team1_home (bool): Whether team1 is playing at home.
            
        Returns:
            float: Probability that team1 beats team2.
        """
        if self.team_encoder is None or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Create feature vector for this matchup
        X = pd.DataFrame([[team1, team2]], columns=['team1', 'team2'])
        
        # Transform using the encoder
        team_encoded = self.team_encoder.transform(X)
        
        # Add additional features if available
        if hasattr(self.feature_scaler, 'mean_'):
            # Create a feature vector with the same number of features as during training
            # Default values: home advantage = 1/0, form = 0.5, home/away form = 0.5, goal_diff = 1, total_goals = 2
            features = np.array([[
                1 if is_team1_home else 0,  # Home advantage
                0.5,                         # Form (neutral)
                0.5,                         # Home/away form (neutral)
                1.0,                         # Goal difference (small)
                2.0                          # Total goals (average)
            ]])
            
            # Scale features
            features = self.feature_scaler.transform(features)
            
            # Combine features
            X_encoded = np.hstack([team_encoded, features])
        else:
            X_encoded = team_encoded
        
        # Predict win probability
        return self.model.predict_proba(X_encoded)[0, 1]
    
    def get_rankings(self, teams):
        """
        Get team rankings based on model coefficients.
        
        Args:
            teams (list): List of team names.
            
        Returns:
            list: Teams sorted by rank (best to worst).
        """
        if self.team_encoder is None or self.model is None:
            raise ValueError("Model must be trained before getting rankings")
        
        # Get team strengths from model predictions
        n_teams = len(teams)
        team_strengths = np.zeros(n_teams)
        
        # For each possible matchup, calculate win probability
        for i, team1 in enumerate(teams):
            for j, team2 in enumerate(teams):
                if i != j:
                    # Predict with home advantage
                    home_win_prob = self.predict_match(team1, team2, is_team1_home=True)
                    away_win_prob = self.predict_match(team1, team2, is_team1_home=False)
                    
                    # Average of home and away probabilities
                    win_prob = (home_win_prob + away_win_prob) / 2
                    
                    # Add to team strength
                    team_strengths[i] += win_prob
        
        # Rank teams by total strength
        ranked_indices = np.argsort(-team_strengths)
        
        return [teams[i] for i in ranked_indices]


class RandomForestModel:
    """
    Random Forest model for predicting match outcomes.
    """
    
    def __init__(self, n_estimators=100, max_depth=None):
        """
        Initialize the Random Forest model.
        
        Args:
            n_estimators (int): Number of trees in the forest.
            max_depth (int): Maximum depth of the trees.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        self.team_encoder = None
        self.feature_scaler = StandardScaler()
        self.team_to_idx = None
    
    def _prepare_data(self, comparisons, team_to_idx):
        """
        Prepare data for training/testing with enhanced features.
        
        Args:
            comparisons (pd.DataFrame): DataFrame with pairwise comparisons.
            team_to_idx (dict): Mapping from team names to indices.
            
        Returns:
            tuple: (X, y) - Features and labels.
        """
        X_teams = []
        X_features = []
        y = []
        
        for _, row in comparisons.iterrows():
            winner = row['winner']
            loser = row['loser']
            
            if winner in team_to_idx and loser in team_to_idx:
                # Skip draws for direct prediction (they're handled differently)
                if 'is_draw' in row and row['is_draw']:
                    continue
                
                # Create team identity features
                X_teams.append([winner, loser])
                y.append(1)  # Winner beats loser
                
                X_teams.append([loser, winner])
                y.append(0)  # Loser doesn't beat winner
                
                # Extract additional features if available
                features1 = []
                features2 = []
                
                # Add home advantage
                if 'home_team' in row and 'away_team' in row:
                    is_home1 = 1 if row['home_team'] == winner else 0
                    is_home2 = 1 if row['home_team'] == loser else 0
                    features1.append(is_home1)
                    features2.append(is_home2)
                else:
                    # Default values if home/away info not available
                    features1.append(0.5)  # Neutral
                    features2.append(0.5)  # Neutral
                
                # Add form
                if 'home_form' in row and 'away_form' in row:
                    if row['home_team'] == winner:
                        form1 = row['home_form']
                        form2 = row['away_form']
                    else:
                        form1 = row['away_form']
                        form2 = row['home_form']
                    features1.append(form1)
                    features2.append(form2)
                else:
                    # Default values if form info not available
                    features1.append(0.5)  # Neutral
                    features2.append(0.5)  # Neutral
                
                # Add home/away specific form
                if 'home_home_form' in row and 'away_away_form' in row and 'home_team' in row:
                    if row['home_team'] == winner:
                        home_form1 = row['home_home_form']
                        away_form2 = row['away_away_form']
                    else:
                        away_form1 = row['away_away_form']
                        home_form2 = row['home_home_form']
                    features1.append(home_form1 if row['home_team'] == winner else away_form1)
                    features2.append(away_form2 if row['home_team'] == winner else home_form2)
                else:
                    # Default values if home/away form info not available
                    features1.append(0.5)  # Neutral
                    features2.append(0.5)  # Neutral
                
                # Add goal-related features
                if 'goal_diff' in row:
                    features1.append(row['goal_diff'])
                    features2.append(row['goal_diff'])
                else:
                    # Default value if goal diff not available
                    features1.append(1.0)  # Small difference
                    features2.append(1.0)  # Small difference
                
                if 'total_goals' in row:
                    features1.append(row['total_goals'])
                    features2.append(row['total_goals'])
                else:
                    # Default value if total goals not available
                    features1.append(2.0)  # Average number of goals
                    features2.append(2.0)  # Average number of goals
                
                # Add features for both comparisons
                X_features.append(features1)
                X_features.append(features2)
        
        # Convert to DataFrame
        X_teams_df = pd.DataFrame(X_teams, columns=['team1', 'team2'])
        
        # One-hot encode teams
        if self.team_encoder is None:
            # For training
            self.team_encoder = OneHotEncoder(sparse_output=False)
            team_encoded = self.team_encoder.fit_transform(X_teams_df)
        else:
            # For testing
            team_encoded = self.team_encoder.transform(X_teams_df)
        
        # Combine with additional features if available
        if X_features and len(X_features[0]) > 0:
            X_features = np.array(X_features)
            
            # Scale features
            if self.feature_scaler is not None:
                if len(X_features) == len(team_encoded):
                    if hasattr(self.model, 'feature_importances_'):  # Model is already trained
                        X_features = self.feature_scaler.transform(X_features)
                    else:  # Model is not trained yet
                        X_features = self.feature_scaler.fit_transform(X_features)
                    
                    # Combine team encoding with additional features
                    X_encoded = np.hstack([team_encoded, X_features])
                else:
                    X_encoded = team_encoded
            else:
                X_encoded = np.hstack([team_encoded, X_features])
        else:
            X_encoded = team_encoded
        
        return X_encoded, np.array(y)
    
    def train(self, comparisons, team_to_idx):
        """
        Train the model.
        
        Args:
            comparisons (pd.DataFrame): DataFrame with pairwise comparisons.
            team_to_idx (dict): Mapping from team names to indices.
            
        Returns:
            self: Trained model.
        """
        self.team_to_idx = team_to_idx
        
        X, y = self._prepare_data(comparisons, team_to_idx)
        
        if len(X) > 0:
            self.model.fit(X, y)
        
        return self
    
    def evaluate(self, test_comparisons, team_to_idx):
        """
        Evaluate model on test data.
        
        Args:
            test_comparisons (pd.DataFrame): DataFrame with test pairwise comparisons.
            team_to_idx (dict): Mapping from team names to indices.
            
        Returns:
            float: Accuracy of predictions.
        """
        # Skip draws for evaluation
        non_draw_comparisons = test_comparisons
        if 'is_draw' in test_comparisons.columns:
            non_draw_comparisons = test_comparisons[~test_comparisons['is_draw']]
        
        X, y = self._prepare_data(non_draw_comparisons, team_to_idx)
        
        if len(X) > 0:
            return self.model.score(X, y)
        else:
            return 0.0
    
    def predict_match(self, team1, team2, is_team1_home=True):
        """
        Predict the outcome of a match between two teams.
        
        Args:
            team1 (str): First team.
            team2 (str): Second team.
            is_team1_home (bool): Whether team1 is playing at home.
            
        Returns:
            float: Probability that team1 beats team2.
        """
        if self.team_encoder is None or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Create feature vector for this matchup
        X = pd.DataFrame([[team1, team2]], columns=['team1', 'team2'])
        
        # Transform using the encoder
        team_encoded = self.team_encoder.transform(X)
        
        # Add additional features if available
        if hasattr(self.feature_scaler, 'mean_'):
            # Create a feature vector with the same number of features as during training
            # Default values: home advantage = 1/0, form = 0.5, home/away form = 0.5, goal_diff = 1, total_goals = 2
            features = np.array([[
                1 if is_team1_home else 0,  # Home advantage
                0.5,                         # Form (neutral)
                0.5,                         # Home/away form (neutral)
                1.0,                         # Goal difference (small)
                2.0                          # Total goals (average)
            ]])
            
            # Scale features
            features = self.feature_scaler.transform(features)
            
            # Combine features
            X_encoded = np.hstack([team_encoded, features])
        else:
            X_encoded = team_encoded
        
        # Predict win probability
        return self.model.predict_proba(X_encoded)[0, 1]
    
    def get_rankings(self, teams):
        """
        Get team rankings based on model predictions.
        
        Args:
            teams (list): List of team names.
            
        Returns:
            list: Teams sorted by rank (best to worst).
        """
        if self.team_encoder is None or self.model is None:
            raise ValueError("Model must be trained before getting rankings")
        
        # Get team strengths from model predictions
        n_teams = len(teams)
        team_strengths = np.zeros(n_teams)
        
        # For each possible matchup, calculate win probability
        for i, team1 in enumerate(teams):
            for j, team2 in enumerate(teams):
                if i != j:
                    # Predict with home advantage
                    home_win_prob = self.predict_match(team1, team2, is_team1_home=True)
                    away_win_prob = self.predict_match(team1, team2, is_team1_home=False)
                    
                    # Average of home and away probabilities
                    win_prob = (home_win_prob + away_win_prob) / 2
                    
                    # Add to team strength
                    team_strengths[i] += win_prob
        
        # Rank teams by total strength
        ranked_indices = np.argsort(-team_strengths)
        
        return [teams[i] for i in ranked_indices] 