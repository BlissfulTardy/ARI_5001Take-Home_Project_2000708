
"""
visualize_training.py - Training Visualization Utilities

Provides detailed step-by-step visualization of agent behavior
for debugging and demonstration purposes.
"""

import numpy as np
import time
import config as cfg
from environment import FootballEnvironment
from agent import ManagerAgent


def print_state_dashboard(step_global, episode, state, agent, valid_actions, env):
    """
    Print detailed dashboard of current state.
    
    Args:
        step_global: Global step counter
        episode: Current episode number
        state: Current state tuple
        agent: Agent instance
        valid_actions: List of valid actions
        env: Environment instance (for additional info)
    """
    t, L, K, C = state
    
    print(f"\n{'=' * 70}")
    print(f" GLOBAL STEP {step_global} | EPISODE {episode} | TURN {t}/{cfg.TURNS_MAX}")
    print(f"{'=' * 70}")
    
    # Player Status Table
    print(f"\n{'ATTR':<8} | {'LEVEL':<6} | {'POT':<4} | {'EXP':<6} | {'STATUS':<10}")
    print("-" * 50)
    
    for i, name in enumerate(cfg.ATTRIBUTES):
        lvl = L[i]
        pot = C[i]
        
        # Get expected level
        if K[i] == 1:
            exp_lvl = lvl
            status = "MAXED"
        else:
            exp_lvl = cfg.EXPECTED_LEVELS.get(pot, {lvl: lvl}).get(lvl, lvl) if pot in cfg.EXPECTED_LEVELS else lvl
            status = "Open"
        
        print(f"{name:<8} | {lvl:<6} | {pot:<4} | {exp_lvl:<6.2f} | {status:<10}")
    
    print("-" * 50)
    
    # Current metrics
    info = env.get_state_info()
    print(f"\nΥ_now: {info['proficiency_now']:.2f} | Υ_future: {info['proficiency_future']:.2f}")
    print(f"δ(t): {info['delta']:.3f} | Value: {info['current_value']:,.0f} | Wage: {info['current_wage']:,.0f}")
    
    # Agent Q-Values
    if state in agent.Q:
        print(f"\n[Agent Q-Values for this State]")
        q_vals = agent.Q[state]
        for act in valid_actions:
            val = q_vals.get(act, 0.0)
            marker = " <--" if val == max(q_vals.get(a, 0) for a in valid_actions) else ""
            print(f"   {act:<12}: {val:>12,.0f}{marker}")
    else:
        print(f"\n[Agent Q-Values]: Unvisited State (Random Exploration)")
    
    print(f"\nValid Actions: {valid_actions}")
    print(f"Current Epsilon: {agent.epsilon:.4f}")


def run_detailed_visualization(num_steps=50, pause=0.0):
    """
    Run detailed step-by-step visualization.
    
    Args:
        num_steps: Maximum number of steps to visualize
        pause: Pause between steps (seconds)
    """
    print("=" * 70)
    print("DETAILED TRAINING VISUALIZATION")
    print("=" * 70)
    
    # Initialize
    env = FootballEnvironment(mode="STOCHASTIC")
    agent = ManagerAgent(algorithm="Q_LEARNING")
    
    global_step = 0
    episode = 1
    
    while global_step < num_steps:
        state = env.reset()
        done = False
        episode_reward = 0
        
        print(f"\n{'#' * 70}")
        print(f"# EPISODE {episode} STARTED")
        print(f"{'#' * 70}")
        
        # Get initial action
        valid_actions = env.get_valid_actions()
        action = agent.get_action(state, valid_actions)
        
        while not done and global_step < num_steps:
            global_step += 1
            
            # Visualize pre-action state
            print_state_dashboard(global_step, episode, state, agent, valid_actions, env)
            
            # Show decision
            print(f"\n>>> AGENT DECISION: {action}")
            
            # Execute action
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Analyze outcome
            if action == "Sell":
                print(f"    RESULT: Sold Player!")
                print(f"    Final Value Reward: {reward:,.0f}")
            else:
                # Determine what happened
                attr_idx = cfg.TRAIN_ACTION_TO_ATTR_IDX.get(action, 0)
                prev_L = state[1][attr_idx]
                prev_K = state[2][attr_idx]
                
                if next_state is not None:
                    curr_L = next_state[1][attr_idx]
                    curr_K = next_state[2][attr_idx]
                    
                    if curr_L > prev_L:
                        print(f"    RESULT: SUCCESS! Level increased {prev_L} -> {curr_L}")
                    elif curr_K > prev_K:
                        print(f"    RESULT: FAILED! Attribute MAXED OUT at level {curr_L}")
                    else:
                        print(f"    RESULT: No change (unexpected)")
                
                print(f"    Wage Cost: {-reward:,.0f}")
            
            # Prepare for update
            next_action = None
            valid_actions_next = []
            
            if not done:
                valid_actions_next = env.get_valid_actions()
                next_action = agent.get_action(next_state, valid_actions_next)
            
            # Update agent
            agent.update(state, action, reward, next_state, next_action, valid_actions_next)
            
            # Transition
            state = next_state
            action = next_action
            valid_actions = valid_actions_next
            
            if pause > 0:
                time.sleep(pause)
        
        # Episode summary
        print(f"\n{'*' * 70}")
        print(f"* EPISODE {episode} FINISHED")
        print(f"* Total Reward: {episode_reward:,.0f}")
        print(f"{'*' * 70}")
        
        agent.decay_epsilon(episode)
        episode += 1
    
    print(f"\n{'=' * 70}")
    print(f"Visualization Complete: {global_step} steps across {episode - 1} episodes")
    print(f"{'=' * 70}")
    
    return agent


def visualize_trained_policy(q_table, num_episodes=5):
    """
    Visualize behavior of a trained policy (greedy).
    
    Args:
        q_table: Trained Q-table
        num_episodes: Number of episodes to visualize
    """
    print("=" * 70)
    print("TRAINED POLICY VISUALIZATION (Greedy)")
    print("=" * 70)
    
    env = FootballEnvironment(mode="STOCHASTIC")
    agent = ManagerAgent(algorithm="Q_LEARNING")
    agent.Q = q_table
    agent.epsilon = 0.0  # Greedy
    
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        turn = 0
        
        print(f"\n{'#' * 70}")
        print(f"# EPISODE {ep}")
        print(f"{'#' * 70}")
        
        # Show initial player
        info = env.get_state_info()
        print(f"\nInitial Player:")
        print(f"  Levels: {list(state[1])}")
        print(f"  Potentials: {list(state[3])}")
        print(f"  Initial Υ: {info['proficiency_now']:.2f}")
        print(f"  Projected Υ: {info['proficiency_future']:.2f}")
        
        trajectory = []
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.get_action(state, valid_actions)
            
            trajectory.append({
                'turn': state[0],
                'action': action,
                'proficiency': env._compute_best_proficiency(env.L)
            })
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            turn += 1
        
        # Show trajectory summary
        print(f"\nTrajectory ({len(trajectory)} actions):")
        for i, step in enumerate(trajectory):
            if i < 5 or i >= len(trajectory) - 3:
                print(f"  t={step['turn']:>2}: {step['action']:<12} (Υ={step['proficiency']:.2f})")
            elif i == 5:
                print(f"  ... ({len(trajectory) - 8} more actions) ...")
        
        print(f"\nFinal Result:")
        print(f"  Final Υ: {trajectory[-1]['proficiency']:.2f}")
        print(f"  Sell Turn: {trajectory[-1]['turn']}")
        print(f"  Total Profit: {total_reward:,.0f}")
    
    return


def compare_algorithms_visual(num_episodes=3):
    """
    Side-by-side comparison of Q-Learning vs SARSA behavior.
    
    Args:
        num_episodes: Number of episodes per algorithm
    """
    print("=" * 70)
    print("ALGORITHM COMPARISON: Q-Learning vs SARSA")
    print("=" * 70)
    
    # Train both briefly
    from training_utils import run_training_session
    
    print("\n--- Training Q-Learning (5000 episodes) ---")
    _, q_table_ql = run_training_session(
        algo="Q_LEARNING",
        num_episodes=5000,
        verbose=False
    )
    
    print("--- Training SARSA (5000 episodes) ---")
    _, q_table_sarsa = run_training_session(
        algo="SARSA",
        num_episodes=5000,
        verbose=False
    )
    
    # Compare on same random seeds
    env = FootballEnvironment(mode="STOCHASTIC")
    
    for ep in range(1, num_episodes + 1):
        # Set same seed for fair comparison
        np.random.seed(42 + ep)
        
        print(f"\n{'=' * 70}")
        print(f"COMPARISON EPISODE {ep}")
        print(f"{'=' * 70}")
        
        for algo_name, q_table in [("Q-Learning", q_table_ql), ("SARSA", q_table_sarsa)]:
            np.random.seed(42 + ep)  # Reset seed
            
            agent = ManagerAgent(algorithm="Q_LEARNING")
            agent.Q = q_table
            agent.epsilon = 0.0
            
            state = env.reset()
            done = False
            total_reward = 0
            actions_taken = []
            
            while not done:
                valid_actions = env.get_valid_actions()
                action = agent.get_action(state, valid_actions)
                actions_taken.append(action)
                
                next_state, reward, done = env.step(action)
                total_reward += reward
                state = next_state
            
            sell_turn = len([a for a in actions_taken if a != "Sell"])
            
            print(f"\n{algo_name}:")
            print(f"  Actions: {len(actions_taken)}")
            print(f"  Sell Turn: {sell_turn}")
            print(f"  Total Profit: {total_reward:,.0f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--trained":
        # Visualize trained policy
        from training_utils import run_training_session
        print("Training agent first...")
        _, q_table = run_training_session(num_episodes=10000, verbose=False)
        visualize_trained_policy(q_table, num_episodes=3)
    elif len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_algorithms_visual()
    else:
        # Default: detailed training visualization
        run_detailed_visualization(num_steps=30)
