
"""
Implementing Q-Learning (Off-Policy) and SARSA (On-Policy) agents
for the Manager-Agent MDP simulation.
"""

import numpy as np
from collections import defaultdict
import config as cfg


class ManagerAgent:
    """
    Reinforcement Learning Agent for Football Player Development.
    
    Supports two algorithms:
    - Q-Learning (Off-Policy): Learns optimal action-value function Q*
    - SARSA (On-Policy): Learns action-value function for current policy
    
    Uses ε-greedy exploration with GLIE (Greedy in Limit with Infinite Exploration).
    """
    
    def __init__(self, algorithm="Q_LEARNING", alpha=None, gamma=None, epsilon=None):
        """
        Initialize the agent.
        
        Args:
            algorithm: "Q_LEARNING" or "SARSA"
            alpha: Learning rate (default: cfg.ALPHA_BASE)
            gamma: Discount factor (default: cfg.GAMMA_BASE)
            epsilon: Initial exploration rate (default: cfg.EPSILON_START)
        """
        if algorithm not in ["Q_LEARNING", "SARSA"]:
            raise ValueError(f"Invalid algorithm: {algorithm}. Must be 'Q_LEARNING' or 'SARSA'")
        
        self.algorithm = algorithm
        self.alpha = alpha if alpha is not None else cfg.ALPHA_BASE
        self.gamma = gamma if gamma is not None else cfg.GAMMA_BASE
        self.epsilon = epsilon if epsilon is not None else cfg.EPSILON_START
        
        # Epsilon decay parameter (can be modified for experiments)
        self.lambda_epsilon = cfg.LAMBDA_EPSILON
        
        # Q-Table: Sparse dictionary mapping state -> {action -> value}
        # Using defaultdict for lazy initialization
        self.Q = defaultdict(lambda: defaultdict(float))
        
        # Statistics tracking
        self.stats = {
            'states_visited': set(),
            'updates_count': 0,
            'episode_count': 0
        }
    
    def get_action(self, state, valid_actions):
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state tuple
            valid_actions: List of valid action strings
            
        Returns:
            Selected action string
        """
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Track visited states
        self.stats['states_visited'].add(state)
        
        # ε-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: Random action
            return np.random.choice(valid_actions)
        
        # Exploit: Greedy action
        return self._get_greedy_action(state, valid_actions)
    
    def _get_greedy_action(self, state, valid_actions):
        """
        Select greedy action (highest Q-value).
        
        Args:
            state: Current state tuple
            valid_actions: List of valid action strings
            
        Returns:
            Action with highest Q-value (random tie-breaking)
        """
        # If state not visited, return random action
        if state not in self.Q:
            return np.random.choice(valid_actions)
        
        # Get Q-values for valid actions only
        q_values = {a: self.Q[state][a] for a in valid_actions}
        
        # Handle empty Q-values
        if not q_values:
            return np.random.choice(valid_actions)
        
        # Find maximum Q-value
        max_q = max(q_values.values())
        
        # Get all actions with maximum Q-value (for tie-breaking)
        best_actions = [a for a, q in q_values.items() if q == max_q]
        
        # Random tie-breaking
        return np.random.choice(best_actions)
    
    def update(self, state, action, reward, next_state, next_action=None, valid_actions_next=None):
        """
        Update Q-value using TD learning.
        
        Implements Algorithm B: UpdateQTable
        
        For Q-Learning (Off-Policy):
            Q(s,a) ← Q(s,a) + α[R + γ·max_a' Q(s',a') - Q(s,a)]
            
        For SARSA (On-Policy):
            Q(s,a) ← Q(s,a) + α[R + γ·Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state tuple
            action: Action taken
            reward: Reward received
            next_state: Resulting state (None if terminal)
            next_action: Action selected in next state (required for SARSA)
            valid_actions_next: Valid actions in next state (required for Q-Learning)
        """
        # Current Q-value
        current_q = self.Q[state][action]
        
        # Compute target value
        if next_state is None:
            # Terminal state: No future value
            target = reward
        else:
            if self.algorithm == "Q_LEARNING":
                # Off-Policy: Use max over valid next actions
                if valid_actions_next:
                    max_next_q = max(self.Q[next_state][a] for a in valid_actions_next)
                else:
                    max_next_q = 0.0
                target = reward + self.gamma * max_next_q
                
            else:  # SARSA
                # On-Policy: Use actual next action
                if next_action is None:
                    # This shouldn't happen in proper SARSA flow
                    next_q = 0.0
                else:
                    next_q = self.Q[next_state][next_action]
                target = reward + self.gamma * next_q
        
        # TD Error
        td_error = target - current_q
        
        # Update Q-value
        self.Q[state][action] = current_q + self.alpha * td_error
        
        # Update statistics
        self.stats['updates_count'] += 1
    
    def decay_epsilon(self, episode):
        """
        Decay exploration rate using harmonic schedule.
        
        ε_k = 1 / (1 + λ × k)
        
        Args:
            episode: Current episode number
        """
        self.epsilon = 1.0 / (1.0 + self.lambda_epsilon * episode)
        self.epsilon = max(self.epsilon, cfg.EPSILON_MIN)
    
    def set_epsilon_decay(self, lambda_val):
        """
        Set epsilon decay parameter.
        
        Args:
            lambda_val: New decay rate
        """
        self.lambda_epsilon = lambda_val
    
    def end_episode(self):
        """Called at the end of each episode for bookkeeping."""
        self.stats['episode_count'] += 1
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def get_q_table_size(self):
        """Get number of states in Q-table."""
        return len(self.Q)
    
    def get_total_entries(self):
        """Get total number of (state, action) pairs in Q-table."""
        total = 0
        for state in self.Q:
            total += len(self.Q[state])
        return total
    
    def get_policy_action(self, state, valid_actions):
        """
        Get the greedy policy action (for evaluation).
        
        Args:
            state: State tuple
            valid_actions: List of valid actions
            
        Returns:
            Greedy action
        """
        return self._get_greedy_action(state, valid_actions)
    
    def get_state_values(self, state, valid_actions):
        """
        Get Q-values for all valid actions in a state.
        
        Args:
            state: State tuple
            valid_actions: List of valid actions
            
        Returns:
            Dictionary {action: Q-value}
        """
        return {a: self.Q[state][a] for a in valid_actions}
    
    def get_statistics(self):
        """Get agent statistics."""
        return {
            'algorithm': self.algorithm,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'lambda_epsilon': self.lambda_epsilon,
            'q_table_states': self.get_q_table_size(),
            'q_table_entries': self.get_total_entries(),
            'unique_states_visited': len(self.stats['states_visited']),
            'total_updates': self.stats['updates_count'],
            'episodes_completed': self.stats['episode_count']
        }
    
    def print_statistics(self):
        """Print agent statistics to console."""
        stats = self.get_statistics()
        
        print(f"\n{'='*50}")
        print(f"AGENT STATISTICS - {stats['algorithm']}")
        print(f"{'='*50}")
        print(f"Learning Rate (α):     {stats['alpha']}")
        print(f"Discount Factor (γ):   {stats['gamma']}")
        print(f"Current Epsilon:       {stats['epsilon']:.4f}")
        print(f"Epsilon Decay (λ):     {stats['lambda_epsilon']}")
        print(f"Q-Table States:        {stats['q_table_states']:,}")
        print(f"Q-Table Entries:       {stats['q_table_entries']:,}")
        print(f"Unique States Seen:    {stats['unique_states_visited']:,}")
        print(f"Total Q-Updates:       {stats['total_updates']:,}")
        print(f"Episodes Completed:    {stats['episodes_completed']:,}")
        print(f"{'='*50}")
    
    # =========================================================================
    # Serialization Methods
    # =========================================================================
    
    def get_q_table_copy(self):
        """
        Get a copy of the Q-table for saving/analysis.
        
        Returns:
            Dictionary representation of Q-table
        """
        # Convert defaultdict to regular dict for serialization
        q_copy = {}
        for state in self.Q:
            q_copy[state] = dict(self.Q[state])
        return q_copy
    
    def load_q_table(self, q_table):
        """
        Load a Q-table from dictionary.
        
        Args:
            q_table: Dictionary {state: {action: value}}
        """
        self.Q = defaultdict(lambda: defaultdict(float))
        for state, actions in q_table.items():
            for action, value in actions.items():
                self.Q[state][action] = value
    
    def reset(self):
        """Reset agent to initial state (clear Q-table and stats)."""
        self.Q = defaultdict(lambda: defaultdict(float))
        self.epsilon = cfg.EPSILON_START
        self.stats = {
            'states_visited': set(),
            'updates_count': 0,
            'episode_count': 0
        }


class AgentFactory:
    """Factory class for creating agents with specific configurations."""
    
    @staticmethod
    def create_q_learning_agent(alpha=None, gamma=None):
        """Create a Q-Learning agent."""
        return ManagerAgent(
            algorithm="Q_LEARNING",
            alpha=alpha,
            gamma=gamma
        )
    
    @staticmethod
    def create_sarsa_agent(alpha=None, gamma=None):
        """Create a SARSA agent."""
        return ManagerAgent(
            algorithm="SARSA",
            alpha=alpha,
            gamma=gamma
        )
    
    @staticmethod
    def create_agent_for_experiment(experiment, **kwargs):
        """
        Create agent configured for specific experiment.
        
        Args:
            experiment: Experiment identifier ('A', 'B1', 'B2', 'C', 'D')
            **kwargs: Additional parameters
            
        Returns:
            Configured ManagerAgent
        """
        if experiment == 'A':
            # Experiment A: Algorithm comparison
            algo = kwargs.get('algorithm', 'Q_LEARNING')
            return ManagerAgent(algorithm=algo)
        
        elif experiment == 'B1':
            # Experiment B1: Learning rate sensitivity
            alpha = kwargs.get('alpha', cfg.ALPHA_BASE)
            return ManagerAgent(algorithm="Q_LEARNING", alpha=alpha)
        
        elif experiment == 'B2':
            # Experiment B2: Exploration schedule
            agent = ManagerAgent(algorithm="Q_LEARNING")
            decay_type = kwargs.get('decay_type', 'Standard')
            agent.lambda_epsilon = cfg.EXP_B2_DECAYS.get(decay_type, cfg.LAMBDA_EPSILON)
            return agent
        
        elif experiment == 'C':
            # Experiment C: Stochastic vs Deterministic
            return ManagerAgent(algorithm="Q_LEARNING")
        
        elif experiment == 'D':
            # Experiment D: Discount factor sensitivity
            gamma = kwargs.get('gamma', cfg.GAMMA_BASE)
            return ManagerAgent(algorithm="Q_LEARNING", gamma=gamma)
        
        else:
            raise ValueError(f"Unknown experiment: {experiment}")


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    from environment import FootballEnvironment
    
    print("Testing ManagerAgent...\n")
    
    # Create environment and agent
    env = FootballEnvironment(mode="STOCHASTIC")
    agent = ManagerAgent(algorithm="Q_LEARNING")
    
    print(f"Algorithm: {agent.algorithm}")
    print(f"Initial Epsilon: {agent.epsilon}")
    
    # Run a few episodes
    num_test_episodes = 100
    total_rewards = []
    
    for ep in range(num_test_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Get initial action
        valid_actions = env.get_valid_actions()
        action = agent.get_action(state, valid_actions)
        
        while not done:
            # Take action
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Get next action (for SARSA compatibility)
            next_action = None
            valid_actions_next = []
            
            if not done:
                valid_actions_next = env.get_valid_actions()
                next_action = agent.get_action(next_state, valid_actions_next)
            
            # Update Q-values
            agent.update(state, action, reward, next_state, next_action, valid_actions_next)
            
            # Transition
            state = next_state
            action = next_action
        
        # End of episode
        agent.decay_epsilon(ep + 1)
        agent.end_episode()
        total_rewards.append(episode_reward)
    
    # Print results_1
    print(f"\nCompleted {num_test_episodes} episodes")
    print(f"Average Reward: {np.mean(total_rewards):,.0f}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    
    agent.print_statistics()
    
    # Test SARSA agent
    print("\n" + "="*50)
    print("Testing SARSA Agent...")
    
    sarsa_agent = AgentFactory.create_sarsa_agent()
    
    for ep in range(100):
        state = env.reset()
        done = False
        
        valid_actions = env.get_valid_actions()
        action = sarsa_agent.get_action(state, valid_actions)
        
        while not done:
            next_state, reward, done = env.step(action)
            
            next_action = None
            valid_actions_next = []
            
            if not done:
                valid_actions_next = env.get_valid_actions()
                next_action = sarsa_agent.get_action(next_state, valid_actions_next)
            
            sarsa_agent.update(state, action, reward, next_state, next_action, valid_actions_next)
            
            state = next_state
            action = next_action
        
        sarsa_agent.decay_epsilon(ep + 1)
        sarsa_agent.end_episode()
    
    sarsa_agent.print_statistics()
