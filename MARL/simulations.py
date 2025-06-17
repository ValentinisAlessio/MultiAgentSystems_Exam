from FootballRL import Environment, Agent
import numpy as np
from tqdm import trange
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List
import logging

np.random.seed(161001)

# Constants
N_TRAIN = 10**6
N_TEST = 10**5
N_SIMULATIONS = 10  # Number of complete simulation runs

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GameStats:
    """Data class to hold game statistics"""
    a_wins: int
    b_wins: int
    draws: int
    terminated_games: int
    avg_match_length: float

@dataclass
class ExperimentConfig:
    """Configuration for experiment runs"""
    train_method: str
    test_method: str
    a_is_random: bool
    b_is_random: bool
    include_draws: bool = False

class FootballTrainer:
    """Main class for training and testing football RL agents"""
    
    def __init__(self, n_train: int = N_TRAIN, n_test: int = N_TEST):
        self.n_train = n_train
        self.n_test = n_test
        self.results = []
    
    def train_agents(self, agent_a: Agent, agent_b: Agent, n_episodes: Optional[int] = None) -> Tuple[Agent, Agent]:
        """Train two agents against each other"""
        if n_episodes is None:
            n_episodes = self.n_train
            
        logger.info(f"Starting training for {n_episodes} episodes...")
        
        with trange(n_episodes, desc="Training Progress", unit="Episode") as tbar:
            for _ in tbar:
                # Store old state and possession before taking actions
                old_state = (agent_a.environment.a_state, agent_a.environment.b_state)
                old_possession = agent_a.environment.possession_a
                
                # Choose actions
                a1_action = agent_a.choose_action()
                a2_action = agent_b.choose_action()
                
                # Execute actions in environment
                agent_a.environment.next_turn((a1_action, a2_action))

                # Update agents with old state, old possession, and actions taken
                agent_a.update_agent(old_state, old_possession, (a1_action, a2_action))
                agent_b.update_agent(old_state, old_possession, (a1_action, a2_action))
                
                self._update_training_progress(tbar, agent_a.environment)

        self._log_training_results(agent_a.environment, n_episodes)
        return agent_a, agent_b

    def test_agents(self, agent_a: Agent, agent_b: Agent, include_draws: bool = True, 
                   n_episodes: Optional[int] = None) -> GameStats:
        """Test two agents and return game statistics"""
        if n_episodes is None:
            n_episodes = self.n_test
            
        logger.info(f"Starting testing for {n_episodes} episodes (draws: {include_draws})...")
        
        draws = 0
        with trange(n_episodes, desc="Testing Progress", unit="episode") as tbar:
            for _ in tbar:
                # Choose actions
                a1_action = agent_a.choose_action()
                a2_action = agent_b.choose_action()
                
                if include_draws and np.random.rand() < 0.1:
                    agent_a.environment.reset_players()
                    draws += 1
                else:
                    # old_state = (agent_a.environment.a_state, agent_a.environment.b_state)
                    # old_possession = agent_a.environment.possession_a
                    
                    # Execute actions
                    agent_a.environment.next_turn((a1_action, a2_action))
                
                self._update_testing_progress(tbar, agent_a.environment, agent_b.environment, draws)

        return self._calculate_game_stats(agent_a.environment, agent_b.environment, draws)

    def run_single_simulation(self, simulation_id: int) -> List[dict]:
        """Run all experiments for a single simulation"""
        logger.info(f"=== STARTING SIMULATION {simulation_id + 1}/{N_SIMULATIONS} ===")
        
        simulation_results = []
        
        # ===== PHASE 1: Train Belief vs Random =====
        logger.info(f"=== SIMULATION {simulation_id + 1} - PHASE 1: Training Belief vs Random ===")
        environment = Environment()
        agent_a_belief = Agent(environment, random_policy=False, is_A=True)
        agent_b_random = Agent(environment, random_policy=True, is_A=False)
        
        self.train_agents(agent_a_belief, agent_b_random)
        
        # Test 1a: Belief A vs Random B without draws
        environment.reset()
        agent_a_belief.exp_rate = 0.0
        stats = self.test_agents(agent_a_belief, agent_b_random, include_draws=False)
        simulation_results.append({
            'simulation_id': simulation_id + 1,
            'Train_method': 'vs_random',
            'Test_method': 'not_include_draws',
            'A_is_random': False,
            'B_is_random': True,
            'A_wins': stats.a_wins,
            'B_wins': stats.b_wins,
            'draws': stats.draws,
            'terminated_games': stats.terminated_games,
            'avg_match_length': stats.avg_match_length
        })
        
        # Test 1b: Belief A vs Random B with draws
        environment.reset()
        stats = self.test_agents(agent_a_belief, agent_b_random, include_draws=True)
        simulation_results.append({
            'simulation_id': simulation_id + 1,
            'Train_method': 'vs_random',
            'Test_method': 'include_draws',
            'A_is_random': False,
            'B_is_random': True,
            'A_wins': stats.a_wins,
            'B_wins': stats.b_wins,
            'draws': stats.draws,
            'terminated_games': stats.terminated_games,
            'avg_match_length': stats.avg_match_length
        })
        
        # ===== PHASE 2: Train Belief vs Belief =====
        logger.info(f"=== SIMULATION {simulation_id + 1} - PHASE 2: Training Belief vs Belief ===")
        environment = Environment()
        agent_a_belief = Agent(environment, random_policy=False, is_A=True)
        agent_b_belief = Agent(environment, random_policy=False, is_A=False)
        
        # Create additional random agents for testing
        agent_ra = Agent(environment, random_policy=True, is_A=False)
        agent_rb = Agent(environment, random_policy=True, is_A=True)
        
        # Train belief agents against each other
        self.train_agents(agent_a_belief, agent_b_belief)
        
        # Test 2a: Belief vs Belief without draws
        environment.reset()
        agent_a_belief.exp_rate = 0.0
        agent_a_belief.exp_rate = 0.0
        stats = self.test_agents(agent_a_belief, agent_b_belief, include_draws=False)
        simulation_results.append({
            'simulation_id': simulation_id + 1,
            'Train_method': 'vs_belief',
            'Test_method': 'not_include_draws',
            'A_is_random': False,
            'B_is_random': False,
            'A_wins': stats.a_wins,
            'B_wins': stats.b_wins,
            'draws': stats.draws,
            'terminated_games': stats.terminated_games,
            'avg_match_length': stats.avg_match_length
        })
        
        # Test 2b: Belief A vs Random B without draws
        environment.reset()
        stats = self.test_agents(agent_a_belief, agent_ra, include_draws=False)
        simulation_results.append({
            'simulation_id': simulation_id + 1,
            'Train_method': 'vs_belief',
            'Test_method': 'not_include_draws',
            'A_is_random': False,
            'B_is_random': True,
            'A_wins': stats.a_wins,
            'B_wins': stats.b_wins,
            'draws': stats.draws,
            'terminated_games': stats.terminated_games,
            'avg_match_length': stats.avg_match_length
        })
        
        # Test 2c: Random A vs Belief B without draws
        environment.reset()
        stats = self.test_agents(agent_rb, agent_b_belief, include_draws=False)
        simulation_results.append({
            'simulation_id': simulation_id + 1,
            'Train_method': 'vs_belief',
            'Test_method': 'not_include_draws',
            'A_is_random': True,
            'B_is_random': False,
            'A_wins': stats.a_wins,
            'B_wins': stats.b_wins,
            'draws': stats.draws,
            'terminated_games': stats.terminated_games,
            'avg_match_length': stats.avg_match_length
        })
        
        # Test 2d: Belief vs Belief with draws
        environment.reset()
        stats = self.test_agents(agent_a_belief, agent_b_belief, include_draws=True)
        simulation_results.append({
            'simulation_id': simulation_id + 1,
            'Train_method': 'vs_belief',
            'Test_method': 'include_draws',
            'A_is_random': False,
            'B_is_random': False,
            'A_wins': stats.a_wins,
            'B_wins': stats.b_wins,
            'draws': stats.draws,
            'terminated_games': stats.terminated_games,
            'avg_match_length': stats.avg_match_length
        })
        
        # Test 2e: Belief A vs Random B with draws
        environment.reset()
        stats = self.test_agents(agent_a_belief, agent_ra, include_draws=True)
        simulation_results.append({
            'simulation_id': simulation_id + 1,
            'Train_method': 'vs_belief',
            'Test_method': 'include_draws',
            'A_is_random': False,
            'B_is_random': True,
            'A_wins': stats.a_wins,
            'B_wins': stats.b_wins,
            'draws': stats.draws,
            'terminated_games': stats.terminated_games,
            'avg_match_length': stats.avg_match_length
        })
        
        # Test 2f: Random A vs Belief B with draws
        environment.reset()
        stats = self.test_agents(agent_rb, agent_b_belief, include_draws=True)
        simulation_results.append({
            'simulation_id': simulation_id + 1,
            'Train_method': 'vs_belief',
            'Test_method': 'include_draws',
            'A_is_random': True,
            'B_is_random': False,
            'A_wins': stats.a_wins,
            'B_wins': stats.b_wins,
            'draws': stats.draws,
            'terminated_games': stats.terminated_games,
            'avg_match_length': stats.avg_match_length
        })
        
        logger.info(f"=== SIMULATION {simulation_id + 1} COMPLETED ===")
        return simulation_results

    def save_results(self, filename: str = 'results/simulations.csv'):
        """Save experiment results to CSV"""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        logger.info(f"Combined results saved to {filename}")

    @staticmethod
    def _update_training_progress(tbar, environment):
        """Update training progress bar"""
        if environment.terminated_games > 0:
            win_rate = 100 * (environment.won_games_A / environment.terminated_games)
        else:
            win_rate = 0
            
        tbar.set_postfix({
            'Home wins': environment.won_games_A,
            'Played games': environment.terminated_games,
            'Home wins %': f"{win_rate:.1f}%"
        })

    @staticmethod
    def _update_testing_progress(tbar, env_a, env_b, draws):
        """Update testing progress bar"""
        tbar.set_postfix({
            'Home wins': env_a.won_games_A,
            'Away wins': env_b.won_games_B,
            'Draws': draws,
            'Played games': env_a.terminated_games
        })

    @staticmethod
    def _calculate_game_stats(env_a, env_b, draws) -> GameStats:
        """Calculate and log game statistics"""
        stats = GameStats(
            a_wins=env_a.won_games_A,
            b_wins=env_b.won_games_B,
            draws=draws,
            terminated_games=env_a.terminated_games,
            avg_match_length=np.mean(env_a.match_length_list) if env_a.match_length_list else 0
        )
        
        logger.info(f"Testing completed - Home: {stats.a_wins}, Away: {stats.b_wins}, Draws: {stats.draws}")
        if stats.terminated_games > 0:
            logger.info(f"Home win rate: {stats.a_wins / stats.terminated_games * 100:.2f}%")
            logger.info(f"Away win rate: {stats.b_wins / stats.terminated_games * 100:.2f}%")
        logger.info(f"Average match length: {stats.avg_match_length:.2f} turns")
        
        return stats

    @staticmethod
    def _log_training_results(environment, n_episodes):
        """Log training completion results"""
        logger.info(f"Training completed after {n_episodes} episodes")
        logger.info(f"Home agent won {environment.won_games_A} out of {environment.terminated_games} games")
        logger.info(f"Away agent won {environment.won_games_B} out of {environment.terminated_games} games")


def run_multiple_simulations(n_simulations: int = N_SIMULATIONS) -> pd.DataFrame:
    """Run multiple complete simulations and combine results"""
    trainer = FootballTrainer()
    all_results = []
    
    logger.info(f"Starting {n_simulations} complete simulations...")
    
    for simulation_id in range(n_simulations):
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING COMPLETE SIMULATION {simulation_id + 1}/{n_simulations}")
        logger.info(f"{'='*60}")
        
        # Run single simulation and get all its results
        simulation_results = trainer.run_single_simulation(simulation_id)
        all_results.extend(simulation_results)
        
        logger.info(f"Simulation {simulation_id + 1} completed. Total experiments so far: {len(all_results)}")
    
    # Store all results in trainer for saving
    trainer.results = all_results
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save combined results
    trainer.save_results()
    
    # Log summary statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL {n_simulations} SIMULATIONS COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Total experiments: {len(all_results)}")
    logger.info(f"Experiments per simulation: {len(all_results) // n_simulations}")
    logger.info(f"Unique experiment types: {df.groupby(['Train_method', 'Test_method', 'A_is_random', 'B_is_random']).size().shape[0]}")
    
    return df


def analyze_combined_results(df: pd.DataFrame) -> None:
    """Analyze the combined results from multiple simulations"""
    logger.info("\n=== COMBINED RESULTS ANALYSIS ===")
    
    # Group by experiment configuration
    grouped = df.groupby(['Train_method', 'Test_method', 'A_is_random', 'B_is_random'])
    
    print("\nSummary Statistics Across All Simulations:")
    print("-" * 80)
    
    for name, group in grouped:
        train_method, test_method, a_random, b_random = name
        print(f"\nConfiguration: Train={train_method}, Test={test_method}, A_random={a_random}, B_random={b_random}")
        print(f"Number of simulations: {len(group)}")
        print(f"A_wins - Mean: {group['A_wins'].mean():.1f}, Std: {group['A_wins'].std():.1f}")
        print(f"B_wins - Mean: {group['B_wins'].mean():.1f}, Std: {group['B_wins'].std():.1f}")
        print(f"Draws - Mean: {group['draws'].mean():.1f}, Std: {group['draws'].std():.1f}")
        print(f"Avg match length - Mean: {group['avg_match_length'].mean():.2f}, Std: {group['avg_match_length'].std():.2f}")


if __name__ == "__main__":
    # Run multiple simulations
    combined_df = run_multiple_simulations(N_SIMULATIONS)
    
    # Analyze results
    analyze_combined_results(combined_df)
    
    # Display basic info about the combined DataFrame
    print(f"\nFinal DataFrame shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")
    print(f"\nFirst few rows:")
    print(combined_df.head())
    
    logger.info("All simulations and analysis completed successfully!")