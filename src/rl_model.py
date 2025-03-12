import numpy as np
import pandas as pd
from tqdm import tqdm

class DuelingBanditRL:
    """
    Enhanced Dueling Bandit Reinforcement Learning model for team ranking.
    
    This model treats teams as arms in a multi-armed bandit problem and uses
    pairwise comparisons (match results) to learn team rankings.
    """
    
    def __init__(self, n_teams, alpha=0.1, exploration_factor=1.0, use_features=True, 
                 feature_learning_rate=0.01, decay_rate=0.9999, min_alpha=0.01,
                 use_thompson_sampling=True, use_double_learning=True):
        """
        Initialize the Dueling Bandit RL model with enhanced features.
        
        Args:
            n_teams (int): Number of teams (arms).
            alpha (float): Initial learning rate.
            exploration_factor (float): Controls exploration vs. exploitation.
            use_features (bool): Whether to use additional features.
            feature_learning_rate (float): Learning rate for feature weights.
            decay_rate (float): Rate at which alpha decays during training.
            min_alpha (float): Minimum value for alpha after decay.
            use_thompson_sampling (bool): Whether to use Thompson sampling for exploration.
            use_double_learning (bool): Whether to use double learning for more stable updates.
        """
        self.n_teams = n_teams
        self.alpha = alpha
        self.initial_alpha = alpha
        self.exploration_factor = exploration_factor
        self.use_features = use_features
        self.feature_learning_rate = feature_learning_rate
        self.decay_rate = decay_rate
        self.min_alpha = min_alpha
        self.use_thompson_sampling = use_thompson_sampling
        self.use_double_learning = use_double_learning
        
        # Initialize team strengths (values) with random values
        self.team_values = np.random.normal(0, 0.1, n_teams)
        
        # For double learning (two separate estimates)
        if self.use_double_learning:
            self.team_values_2 = np.random.normal(0, 0.1, n_teams)
        
        # Initialize count of matches for each team
        self.team_counts = np.zeros(n_teams)
        
        # For Thompson sampling
        if self.use_thompson_sampling:
            # Initialize Beta distribution parameters for each team
            self.alpha_params = np.ones(n_teams) * 2  # Prior wins + 1
            self.beta_params = np.ones(n_teams) * 2   # Prior losses + 1
        
        # Initialize UCB values
        self.ucb_values = np.copy(self.team_values)
        
        # Track history of team values for analysis
        self.value_history = [np.copy(self.team_values)]
        
        # Feature weights (for home advantage, form, etc.)
        if use_features:
            # [home_adv, form, home_form, goal_diff, total_goals]
            self.feature_weights = np.random.normal(0, 0.1, 5)
            
            # Track feature weight history
            self.feature_weight_history = [np.copy(self.feature_weights)]
            
            # For double learning
            if self.use_double_learning:
                self.feature_weights_2 = np.random.normal(0, 0.1, 5)
        
        # Training iteration counter
        self.iterations = 0
        
    def update(self, winner_idx, loser_idx, features=None, is_draw=False):
        """
        Update team values based on match outcome with enhanced learning.
        
        Args:
            winner_idx (int): Index of the winning team.
            loser_idx (int): Index of the losing team.
            features (dict): Additional features for the match.
            is_draw (bool): Whether the match was a draw.
        """
        # Update counts
        self.team_counts[winner_idx] += 1
        self.team_counts[loser_idx] += 1
        
        # Update Thompson sampling parameters
        if self.use_thompson_sampling:
            if is_draw:
                # For draws, update both teams' parameters slightly
                self.alpha_params[winner_idx] += 0.5
                self.beta_params[loser_idx] += 0.5
                self.alpha_params[loser_idx] += 0.5
                self.beta_params[winner_idx] += 0.5
            else:
                # Winner gets alpha update, loser gets beta update
                self.alpha_params[winner_idx] += 1
                self.beta_params[loser_idx] += 1
        
        # Calculate reward based on outcome
        if is_draw:
            reward = 0.5  # Draw is a partial reward
        else:
            reward = 1.0  # Win is a full reward
        
        # Adjust reward based on features
        feature_adjustment = 0
        if self.use_features and features is not None:
            # Use either primary or secondary feature weights based on iteration
            weights = self.feature_weights if not self.use_double_learning or self.iterations % 2 == 0 else self.feature_weights_2
            
            # Home advantage
            if 'is_home' in features:
                feature_adjustment += weights[0] * features['is_home']
            
            # Form
            if 'form_diff' in features:
                feature_adjustment += weights[1] * features['form_diff']
            
            # Home/away form
            if 'home_away_form_diff' in features:
                feature_adjustment += weights[2] * features['home_away_form_diff']
            
            # Goal difference
            if 'goal_diff' in features:
                feature_adjustment += weights[3] * features['goal_diff'] / 5.0  # Normalize
            
            # Total goals
            if 'total_goals' in features:
                feature_adjustment += weights[4] * features['total_goals'] / 5.0  # Normalize
            
            # Apply adjustment (capped to avoid extreme values)
            adjusted_reward = max(0.1, min(1.0, reward + feature_adjustment))
            
            # Update feature weights based on prediction error
            prediction_error = reward - adjusted_reward
            
            # Update feature weights
            if 'is_home' in features:
                self.feature_weights[0] += self.feature_learning_rate * prediction_error * features['is_home']
            if 'form_diff' in features:
                self.feature_weights[1] += self.feature_learning_rate * prediction_error * features['form_diff']
            if 'home_away_form_diff' in features:
                self.feature_weights[2] += self.feature_learning_rate * prediction_error * features['home_away_form_diff']
            if 'goal_diff' in features:
                self.feature_weights[3] += self.feature_learning_rate * prediction_error * features['goal_diff'] / 5.0
            if 'total_goals' in features:
                self.feature_weights[4] += self.feature_learning_rate * prediction_error * features['total_goals'] / 5.0
            
            # Track feature weight history
            self.feature_weight_history.append(np.copy(self.feature_weights))
            
            # Use adjusted reward
            reward = adjusted_reward
        
        # Decay learning rate
        self.alpha = max(self.min_alpha, self.alpha * self.decay_rate)
        
        # Double learning update
        if self.use_double_learning:
            if self.iterations % 2 == 0:
                # Update primary values using secondary values for target
                target_winner = reward + self.team_values_2[winner_idx]
                target_loser = (1 - reward) + self.team_values_2[loser_idx]
                
                # Update primary values
                self.team_values[winner_idx] += self.alpha * (target_winner - self.team_values[winner_idx])
                self.team_values[loser_idx] += self.alpha * (target_loser - self.team_values[loser_idx])
            else:
                # Update secondary values using primary values for target
                target_winner = reward + self.team_values[winner_idx]
                target_loser = (1 - reward) + self.team_values[loser_idx]
                
                # Update secondary values
                self.team_values_2[winner_idx] += self.alpha * (target_winner - self.team_values_2[winner_idx])
                self.team_values_2[loser_idx] += self.alpha * (target_loser - self.team_values_2[loser_idx])
        else:
            # Standard TD learning update
            # Winner's value increases
            self.team_values[winner_idx] += self.alpha * (reward - self.team_values[winner_idx])
            
            # Loser's value decreases
            self.team_values[loser_idx] -= self.alpha * (reward * self.team_values[loser_idx])
        
        # Update UCB values
        self._update_ucb()
        
        # Track history
        self.value_history.append(np.copy(self.team_values))
        
        # Increment iteration counter
        self.iterations += 1
    
    def _update_ucb(self):
        """
        Update Upper Confidence Bound values for exploration.
        """
        # Avoid division by zero
        counts = np.maximum(self.team_counts, 1)
        
        # Calculate UCB values
        exploration_term = self.exploration_factor * np.sqrt(2 * np.log(np.sum(counts)) / counts)
        self.ucb_values = self.team_values + exploration_term
    
    def select_match(self):
        """
        Select the next match to simulate based on exploration strategy.
        
        Returns:
            tuple: (team1_idx, team2_idx) - Indices of teams to match.
        """
        if self.use_thompson_sampling:
            # Sample from Beta distributions
            samples = np.random.beta(self.alpha_params, self.beta_params)
            
            # Select team with highest sample
            team1_idx = np.argmax(samples)
            
            # Remove selected team and find second team
            temp_samples = np.copy(samples)
            temp_samples[team1_idx] = -np.inf
            team2_idx = np.argmax(temp_samples)
        else:
            # Use UCB strategy
            # Select team with highest UCB value
            team1_idx = np.argmax(self.ucb_values)
            
            # Remove selected team and find second team
            temp_ucb = np.copy(self.ucb_values)
            temp_ucb[team1_idx] = -np.inf
            team2_idx = np.argmax(temp_ucb)
        
        return team1_idx, team2_idx
    
    def simulate_match(self, team1_idx, team2_idx, preference_matrix):
        """
        Simulate a match between two teams based on preference matrix.
        
        Args:
            team1_idx (int): Index of first team.
            team2_idx (int): Index of second team.
            preference_matrix (np.ndarray): Matrix of win probabilities.
            
        Returns:
            tuple: (winner_idx, loser_idx, is_draw) - Indices of winner and loser, and draw flag.
        """
        # Get win probability from preference matrix
        p_team1_wins = preference_matrix[team1_idx, team2_idx]
        
        # Simulate match outcome with possibility of draw
        rand = np.random.random()
        if rand < p_team1_wins - 0.15:  # Reduce win prob to account for draws
            return team1_idx, team2_idx, False
        elif rand > p_team1_wins + 0.15:  # Increase loss prob to account for draws
            return team2_idx, team1_idx, False
        else:
            # It's a draw, randomly select "winner" for update purposes
            if np.random.random() < 0.5:
                return team1_idx, team2_idx, True
            else:
                return team2_idx, team1_idx, True
    
    def get_rankings(self):
        """
        Get current team rankings based on learned values.
        
        Returns:
            np.ndarray: Indices of teams sorted by rank (best to worst).
        """
        # If using double learning, average the two value estimates
        if self.use_double_learning:
            combined_values = (self.team_values + self.team_values_2) / 2
            return np.argsort(-combined_values)
        else:
            return np.argsort(-self.team_values)
    
    def train(self, preference_matrix, n_iterations=10000):
        """
        Train the model by simulating matches.
        
        Args:
            preference_matrix (np.ndarray): Matrix of win probabilities.
            n_iterations (int): Number of matches to simulate.
            
        Returns:
            np.ndarray: Final team rankings.
        """
        for _ in tqdm(range(n_iterations), desc="Training RL model"):
            # Select match
            team1_idx, team2_idx = self.select_match()
            
            # Simulate match
            winner_idx, loser_idx, is_draw = self.simulate_match(team1_idx, team2_idx, preference_matrix)
            
            # Update values
            self.update(winner_idx, loser_idx, is_draw=is_draw)
        
        return self.get_rankings()
    
    def train_with_real_matches(self, comparisons, team_to_idx):
        """
        Train the model using real match data.
        
        Args:
            comparisons (pd.DataFrame): DataFrame with pairwise comparisons.
            team_to_idx (dict): Mapping from team names to indices.
            
        Returns:
            np.ndarray: Final team rankings.
        """
        # Reset model for fresh training
        self.__init__(
            n_teams=self.n_teams, 
            alpha=self.initial_alpha, 
            exploration_factor=self.exploration_factor,
            use_features=self.use_features,
            feature_learning_rate=self.feature_learning_rate,
            decay_rate=self.decay_rate,
            min_alpha=self.min_alpha,
            use_thompson_sampling=self.use_thompson_sampling,
            use_double_learning=self.use_double_learning
        )
        
        # Shuffle comparisons to avoid bias from chronological order
        shuffled_comparisons = comparisons.sample(frac=1, random_state=42).reset_index(drop=True)
        
        for _, row in tqdm(shuffled_comparisons.iterrows(), total=len(shuffled_comparisons), desc="Training with real matches"):
            winner = row['winner']
            loser = row['loser']
            
            if winner in team_to_idx and loser in team_to_idx:
                winner_idx = team_to_idx[winner]
                loser_idx = team_to_idx[loser]
                
                # Extract features if available
                features = None
                if self.use_features:
                    features = {}
                    
                    # Home advantage
                    if 'home_team' in row and row['home_team'] == winner:
                        features['is_home'] = 1
                    elif 'home_team' in row:
                        features['is_home'] = -1
                    
                    # Form difference
                    if 'home_form' in row and 'away_form' in row:
                        if row['home_team'] == winner:
                            features['form_diff'] = row['home_form'] - row['away_form']
                        else:
                            features['form_diff'] = row['away_form'] - row['home_form']
                    
                    # Home/away form difference
                    if 'home_home_form' in row and 'away_away_form' in row:
                        if row['home_team'] == winner:
                            features['home_away_form_diff'] = row['home_home_form'] - row['away_away_form']
                        else:
                            features['home_away_form_diff'] = row['away_away_form'] - row['home_home_form']
                    
                    # Goal difference
                    if 'goal_diff' in row:
                        features['goal_diff'] = row['goal_diff']
                    
                    # Total goals
                    if 'total_goals' in row:
                        features['total_goals'] = row['total_goals']
                
                # Check if it's a draw
                is_draw = False
                if 'is_draw' in row:
                    is_draw = row['is_draw']
                
                # Update values
                self.update(winner_idx, loser_idx, features=features, is_draw=is_draw)
        
        # Train for a few more iterations with the learned feature weights
        # This helps stabilize the rankings
        if self.use_features:
            # Create a simple preference matrix based on current values
            pref_matrix = np.zeros((self.n_teams, self.n_teams))
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i != j:
                        # Sigmoid of difference in team values
                        if self.use_double_learning:
                            val_i = (self.team_values[i] + self.team_values_2[i]) / 2
                            val_j = (self.team_values[j] + self.team_values_2[j]) / 2
                            pref_matrix[i, j] = 1 / (1 + np.exp(-(val_i - val_j)))
                        else:
                            pref_matrix[i, j] = 1 / (1 + np.exp(-(self.team_values[i] - self.team_values[j])))
            
            # Train with simulated matches using learned preferences
            for _ in tqdm(range(1000), desc="Fine-tuning RL model"):
                team1_idx, team2_idx = self.select_match()
                winner_idx, loser_idx, is_draw = self.simulate_match(team1_idx, team2_idx, pref_matrix)
                self.update(winner_idx, loser_idx, is_draw=is_draw)
        
        return self.get_rankings()
    
    def evaluate(self, test_comparisons, team_to_idx):
        """
        Evaluate model on test data with enhanced prediction.
        
        Args:
            test_comparisons (pd.DataFrame): DataFrame with test pairwise comparisons.
            team_to_idx (dict): Mapping from team names to indices.
            
        Returns:
            float: Accuracy of predictions.
        """
        correct = 0
        total = 0
        
        for _, row in test_comparisons.iterrows():
            winner = row['winner']
            loser = row['loser']
            
            # Skip draws for evaluation
            if 'is_draw' in row and row['is_draw']:
                continue
            
            if winner in team_to_idx and loser in team_to_idx:
                winner_idx = team_to_idx[winner]
                loser_idx = team_to_idx[loser]
                
                # Get base team strengths
                if self.use_double_learning:
                    team1_strength = (self.team_values[winner_idx] + self.team_values_2[winner_idx]) / 2
                    team2_strength = (self.team_values[loser_idx] + self.team_values_2[loser_idx]) / 2
                else:
                    team1_strength = self.team_values[winner_idx]
                    team2_strength = self.team_values[loser_idx]
                
                # Adjust for features if available
                if self.use_features:
                    # Home advantage
                    if 'home_team' in row and row['home_team'] == winner:
                        team1_strength += self.feature_weights[0] * 0.1
                    elif 'home_team' in row and row['home_team'] == loser:
                        team2_strength += self.feature_weights[0] * 0.1
                    
                    # Form
                    if 'home_form' in row and 'away_form' in row:
                        if row['home_team'] == winner:
                            form_diff = row['home_form'] - row['away_form']
                            team1_strength += self.feature_weights[1] * form_diff * 0.05
                        else:
                            form_diff = row['away_form'] - row['home_form']
                            team1_strength += self.feature_weights[1] * form_diff * 0.05
                    
                    # Home/away form
                    if 'home_home_form' in row and 'away_away_form' in row:
                        if row['home_team'] == winner:
                            home_away_diff = row['home_home_form'] - row['away_away_form']
                            team1_strength += self.feature_weights[2] * home_away_diff * 0.05
                        else:
                            home_away_diff = row['away_away_form'] - row['home_home_form']
                            team1_strength += self.feature_weights[2] * home_away_diff * 0.05
                    
                    # Goal difference and total goals can't be known before the match
                
                # Predict winner based on adjusted strengths
                predicted_winner = winner_idx if team1_strength > team2_strength else loser_idx
                
                # Check if prediction is correct
                if predicted_winner == winner_idx:
                    correct += 1
                
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def get_feature_importance(self):
        """
        Get the importance of each feature based on learned weights.
        
        Returns:
            dict: Feature importance scores.
        """
        if not self.use_features:
            return {}
        
        # Normalize weights to get relative importance
        abs_weights = np.abs(self.feature_weights)
        total = np.sum(abs_weights)
        if total == 0:
            normalized = abs_weights
        else:
            normalized = abs_weights / total
        
        return {
            'home_advantage': normalized[0],
            'form': normalized[1],
            'home_away_form': normalized[2],
            'goal_difference': normalized[3],
            'total_goals': normalized[4]
        }


class BradleyTerryModel:
    """
    Bradley-Terry model for team ranking.
    
    This is a traditional model for ranking based on pairwise comparisons.
    """
    
    def __init__(self, n_teams, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the Bradley-Terry model.
        
        Args:
            n_teams (int): Number of teams.
            learning_rate (float): Learning rate for gradient descent.
            n_iterations (int): Number of iterations for training.
        """
        self.n_teams = n_teams
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
        # Initialize team strengths (log-skills)
        self.log_skills = np.zeros(n_teams)
    
    def train(self, comparisons, team_to_idx):
        """
        Train the model using gradient descent.
        
        Args:
            comparisons (pd.DataFrame): DataFrame with pairwise comparisons.
            team_to_idx (dict): Mapping from team names to indices.
            
        Returns:
            np.ndarray: Final team rankings.
        """
        # Create win matrix
        win_matrix = np.zeros((self.n_teams, self.n_teams))
        
        for _, row in comparisons.iterrows():
            winner = row['winner']
            loser = row['loser']
            
            if winner in team_to_idx and loser in team_to_idx:
                winner_idx = team_to_idx[winner]
                loser_idx = team_to_idx[loser]
                
                win_matrix[winner_idx, loser_idx] += 1
        
        # Gradient descent
        for _ in tqdm(range(self.n_iterations), desc="Training Bradley-Terry model"):
            gradients = np.zeros(self.n_teams)
            
            for i in range(self.n_teams):
                for j in range(self.n_teams):
                    if i != j and (win_matrix[i, j] > 0 or win_matrix[j, i] > 0):
                        # Number of matches between i and j
                        n_ij = win_matrix[i, j] + win_matrix[j, i]
                        
                        # Probability that i beats j according to current model
                        p_ij = 1 / (1 + np.exp(self.log_skills[j] - self.log_skills[i]))
                        
                        # Gradient
                        gradients[i] += win_matrix[i, j] - n_ij * p_ij
            
            # Update log-skills
            self.log_skills += self.learning_rate * gradients
            
            # Normalize to prevent drift
            self.log_skills -= np.mean(self.log_skills)
        
        return self.get_rankings()
    
    def get_rankings(self):
        """
        Get current team rankings based on learned skills.
        
        Returns:
            np.ndarray: Indices of teams sorted by rank (best to worst).
        """
        return np.argsort(-self.log_skills)
    
    def evaluate(self, test_comparisons, team_to_idx):
        """
        Evaluate model on test data.
        
        Args:
            test_comparisons (pd.DataFrame): DataFrame with test pairwise comparisons.
            team_to_idx (dict): Mapping from team names to indices.
            
        Returns:
            float: Accuracy of predictions.
        """
        correct = 0
        total = 0
        
        for _, row in test_comparisons.iterrows():
            winner = row['winner']
            loser = row['loser']
            
            if winner in team_to_idx and loser in team_to_idx:
                winner_idx = team_to_idx[winner]
                loser_idx = team_to_idx[loser]
                
                # Predict winner based on current skills
                predicted_winner = winner_idx if self.log_skills[winner_idx] > self.log_skills[loser_idx] else loser_idx
                
                # Check if prediction is correct
                if predicted_winner == winner_idx:
                    correct += 1
                
                total += 1
        
        return correct / total if total > 0 else 0.0 