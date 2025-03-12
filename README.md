# Reinforcement Learning for Team Ranking in Eliteserien

This project implements a reinforcement learning approach to rank football teams in the Norwegian Eliteserien league using pairwise comparisons within a dueling bandit framework.

## Project Overview

The goal is to recover the true ranking of teams in the Eliteserien football league using historical match data from 2019-2023, and evaluate the model's predictions against the actual 2024 season results.

## Key Features

- Implements a dueling bandit RL model for team ranking
- Handles promotion/relegation dynamics in the league
- Compares RL performance against traditional ranking models
- Evaluates ranking accuracy against actual 2024 standings

## Project Structure

- `data/`: Contains raw and processed match data
- `src/`: Source code for the project
  - `data_processing.py`: Functions for data preprocessing
  - `rl_model.py`: Implementation of the RL ranking model
  - `traditional_models.py`: Implementation of baseline models
  - `evaluation.py`: Functions for evaluating model performance
- `notebooks/`: Jupyter notebooks for analysis and visualization
- `main.py`: Main script to run the entire pipeline

## Setup and Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```
   python main.py
   ```

## Data

The project uses match data from the Norwegian Eliteserien football league from 2019-2024, including:
- Home team, Away team
- Final scores
- Match outcomes

## Methodology

The project implements a dueling bandit RL approach where:
- Teams are treated as arms (options)
- Match results are pairwise comparisons
- UCB-based exploration-exploitation strategy is used to improve ranking accuracy 