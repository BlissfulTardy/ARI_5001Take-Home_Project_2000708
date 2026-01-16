"""
training_utils.py - Training Utilities Module

Provides functions for training agents, running experiments,
and collecting metrics including Profit Efficiency (PE).
"""

import numpy as np
import time
from collections import defaultdict
import config as cfg
from environment import FootballEnvironment
from agent import ManagerAgent


def calculate_theoretical_max_profit(player_class):
    """
    Returns the theoretical maximum profit for a given class (L, M, H)
    assuming perfect training (no fails) and optimal selling time.
    """
    limit = cfg.MAP_LIMIT_DET.get(player_class, 10)
    max_turns = cfg.TURNS_MAX
    best_profit = -float('inf')

    for t in range(1, max_turns + 1):
        remaining_points = t
        temp_levels = []
        for _ in range(5):
            points_to_add = min(limit, remaining_points)
            temp_levels.append(points_to_add)
            remaining_points -= points_to_add

        temp_levels.sort(reverse=True)
        weights = sorted([0.6, 0.2, 0.2, 0.1, 0.0], reverse=True)
        proficiency = sum(w * l for w, l in zip(weights, temp_levels))
        value = cfg.PRICE_BASE * (cfg.BETA_PROF ** proficiency)

        cost = 0
        current_lvl_sum = 0
        for step in range(1, t + 1):
            current_lvl_sum += 1
            cost += (cfg.WAGE_BASE + cfg.PSI_MATURITY * current_lvl_sum) * (1 + cfg.PHI_INFLATION * step)

        profit = value - cost
        if profit > best_profit:
            best_profit = profit

    return best_profit


# Pre-calculate benchmarks once (including B for deterministic mode)
BENCHMARK_PROFITS = {
    'L': calculate_theoretical_max_profit('L'),
    'M': calculate_theoretical_max_profit('M'),
    'H': calculate_theoretical_max_profit('H'),
    'B': calculate_theoretical_max_profit('B')  # Bust class for deterministic
}


def analyze_efficiency(rewards_history, quality_history):
    """
    Calculates Profit Efficiency (PE).
    PE = average(actual_profit / theoretical_max_profit) for each episode.
    """
    if not rewards_history or not quality_history:
        return 0.0

    efficiencies = []
    for reward, p_class in zip(rewards_history, quality_history):
        max_possible = BENCHMARK_PROFITS.get(p_class, 1.0)
        if max_possible == 0 or max_possible is None:
            max_possible = 1.0
        eff = reward / max_possible
        efficiencies.append(eff)

    return np.mean(efficiencies)


def run_training_session(
    env_mode="STOCHASTIC",
    algo="Q_LEARNING",
    alpha=None,
    gamma=None,
    decay_lambda=None,
    num_episodes=None,
    label="Training",
    verbose=True,
    log_interval=cfg.LOG_INTERVAL
):
    """
    Run a complete training session with PE tracking.

    Returns:
        Tuple (rewards_history, q_table)
    """
    alpha = alpha if alpha is not None else cfg.ALPHA_BASE
    gamma = gamma if gamma is not None else cfg.GAMMA_BASE
    decay_lambda = decay_lambda if decay_lambda is not None else cfg.LAMBDA_EPSILON
    num_episodes = num_episodes if num_episodes is not None else cfg.NUM_EPISODES

    if verbose:
        print(f"\n{'='*60}")
        print(f"TRAINING SESSION: {label}")
        print(f"{'='*60}")
        print(f"  Mode:       {env_mode}")
        print(f"  Algorithm:  {algo}")
        print(f"  Alpha:      {alpha}")
        print(f"  Gamma:      {gamma}")
        print(f"  Lambda:     {decay_lambda}")
        print(f"  Episodes:   {num_episodes:,}")
        print(f"{'='*60}")

    env = FootballEnvironment(mode=env_mode)
    agent = ManagerAgent(algorithm=algo, alpha=alpha, gamma=gamma)
    agent.lambda_epsilon = decay_lambda

    rewards_history = []
    player_classes_history = []
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        player_class = state[3][0]
        player_classes_history.append(player_class)

        episode_reward = 0
        done = False

        valid_actions = env.get_valid_actions()
        action = agent.get_action(state, valid_actions)

        while not done:
            next_state, reward, done = env.step(action)
            episode_reward += reward

            next_action = None
            valid_actions_next = []

            if not done:
                valid_actions_next = env.get_valid_actions()
                next_action = agent.get_action(next_state, valid_actions_next)

            agent.update(state, action, reward, next_state, next_action, valid_actions_next)
            state = next_state
            action = next_action

        agent.decay_epsilon(episode)
        agent.end_episode()
        rewards_history.append(episode_reward)

        if verbose and episode % log_interval == 0:
            avg_reward = np.mean(rewards_history[-1000:])
            recent_pe = analyze_efficiency(
                rewards_history[-1000:],
                player_classes_history[-1000:]
            )
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed

            print(f"  Ep {episode:>6,}: Avg Profit = {avg_reward:>12,.0f} | "
                  f"PE = {recent_pe:>6.1%} | "
                  f"eps = {agent.epsilon:.4f} | "
                  f"States = {agent.get_q_table_size():,} | "
                  f"{eps_per_sec:.0f} ep/s")

    elapsed = time.time() - start_time
    final_pe = analyze_efficiency(rewards_history[-1000:], player_classes_history[-1000:])

    if verbose:
        print(f"\n  Training Complete!")
        print(f"  Total Time: {elapsed:.1f}s")
        print(f"  Final Avg Profit (last 1k): {np.mean(rewards_history[-1000:]):,.0f}")
        print(f"  Final PE (last 1k): {final_pe:.1%}")
        print(f"  Q-Table States: {agent.get_q_table_size():,}")
        print(f"{'='*60}\n")

    return rewards_history, agent.Q


def evaluate_policy(
        q_table,
        env_mode="STOCHASTIC",
        num_episodes=1000,
        verbose=True
):
    """
    Evaluate a trained policy (greedy, no exploration) and calculate PE.
    """
    env = FootballEnvironment(mode=env_mode)
    agent = ManagerAgent(algorithm="Q_LEARNING")
    agent.Q = q_table
    agent.epsilon = 0.0

    rewards = []
    sell_times = []
    final_proficiencies = []
    quality_history = []

    for _ in range(num_episodes):
        state = env.reset()
        player_class = state[3][0]
        quality_history.append(player_class)

        episode_profit = 0
        done = False
        sell_time = cfg.TURNS_MAX

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.get_action(state, valid_actions)

            if action == "Sell":
                sell_time = state[0]

            next_state, reward, done = env.step(action)
            episode_profit += reward
            state = next_state

        rewards.append(episode_profit)
        sell_times.append(sell_time)

        if env.L is not None:
            final_prof = env._compute_best_proficiency(env.L)
            final_proficiencies.append(final_prof)

    pe_score = analyze_efficiency(rewards, quality_history)

    metrics = {
        'mean_profit': np.mean(rewards),
        'std_profit': np.std(rewards),
        'min_profit': np.min(rewards),
        'max_profit': np.max(rewards),
        'median_profit': np.median(rewards),
        'mean_sell_time': np.mean(sell_times),
        'std_sell_time': np.std(sell_times),
        'mean_final_proficiency': np.mean(final_proficiencies) if final_proficiencies else 0,
        'pe_score': pe_score,
        'profits': rewards,
        'sell_times': sell_times
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print("EVALUATION RESULTS")
        print(f"{'=' * 50}")
        print(f"  Episodes:        {num_episodes:,}")
        print(f"  Mean Profit:     {metrics['mean_profit']:>12,.0f}")
        print(f"  Std Profit:      {metrics['std_profit']:>12,.0f}")
        print(f"  PE Score:        {metrics['pe_score']:>12.1%}")
        print(f"  Min Profit:      {metrics['min_profit']:>12,.0f}")
        print(f"  Max Profit:      {metrics['max_profit']:>12,.0f}")
        print(f"  Mean Sell Time:  {metrics['mean_sell_time']:>12.1f}")
        print(f"  Mean Final Prof: {metrics['mean_final_proficiency']:>12.2f}")
        print(f"{'=' * 50}\n")

    return metrics


def run_episode_with_tracking(env, agent, track_details=False):
    """Run a single episode with optional detailed tracking."""
    state = env.reset()

    episode_data = {
        'initial_state': state,
        'total_reward': 0,
        'steps': 0,
        'sell_turn': None,
        'trajectory': [] if track_details else None
    }

    done = False
    valid_actions = env.get_valid_actions()
    action = agent.get_action(state, valid_actions)

    while not done:
        if track_details:
            step_info = {
                'turn': state[0],
                'state': state,
                'action': action,
                'q_values': dict(agent.Q[state]) if state in agent.Q else {}
            }

        if action == "Sell":
            episode_data['sell_turn'] = state[0]

        next_state, reward, done = env.step(action)
        episode_data['total_reward'] += reward
        episode_data['steps'] += 1

        if track_details:
            step_info['reward'] = reward
            step_info['done'] = done
            episode_data['trajectory'].append(step_info)

        next_action = None
        valid_actions_next = []

        if not done:
            valid_actions_next = env.get_valid_actions()
            next_action = agent.get_action(next_state, valid_actions_next)

        agent.update(state, action, reward, next_state, next_action, valid_actions_next)

        state = next_state
        action = next_action

    if episode_data['sell_turn'] is None:
        episode_data['sell_turn'] = cfg.TURNS_MAX

    episode_data['final_state'] = env._get_state_tuple() if env.L is not None else None

    return episode_data


def compute_moving_average(values, window=1000):
    """Compute moving average of a sequence."""
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode='valid')


def detect_convergence(rewards, window=1000, threshold=0.05):
    """Detect convergence point in reward sequence."""
    if len(rewards) < window * 2:
        return False, None

    ma = compute_moving_average(rewards, window)
    final_avg = ma[-1]
    stability_margin = abs(final_avg * threshold)

    for i in range(len(ma)):
        if np.all(np.abs(ma[i:] - final_avg) <= stability_margin):
            return True, i + window

    return False, None


def compute_cumulative_regret(rewards, optimal_reward=None):
    """Compute cumulative regret."""
    if optimal_reward is None:
        if len(rewards) > 1000:
            optimal_reward = np.mean(sorted(rewards)[-100:])
        else:
            optimal_reward = max(rewards)

    regret_per_episode = optimal_reward - np.array(rewards)
    cumulative_regret = np.cumsum(regret_per_episode)

    return cumulative_regret


def analyze_policy_by_player_quality(q_table, env_mode="STOCHASTIC", num_episodes=1000):
    """Analyze policy performance segmented by player quality."""
    env = FootballEnvironment(mode=env_mode)
    agent = ManagerAgent(algorithm="Q_LEARNING")
    agent.Q = q_table
    agent.epsilon = 0.0

    results_by_quality = defaultdict(list)
    sell_times_by_quality = defaultdict(list)

    for _ in range(num_episodes):
        state = env.reset()

        potentials = state[3]
        if env_mode == "STOCHASTIC":
            quality = sum(1 for p in potentials if p == 'H') - sum(1 for p in potentials if p == 'L')
        else:
            quality = sum(1 for p in potentials if p == 'H') - sum(1 for p in potentials if p in ['L', 'B'])

        episode_profit = 0
        sell_time = cfg.TURNS_MAX
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.get_action(state, valid_actions)

            if action == "Sell":
                sell_time = state[0]

            next_state, reward, done = env.step(action)
            episode_profit += reward
            state = next_state

        results_by_quality[quality].append(episode_profit)
        sell_times_by_quality[quality].append(sell_time)

    analysis = {
        'by_quality': {},
        'summary': {
            'low_quality': {'profit': [], 'sell_time': []},
            'avg_quality': {'profit': [], 'sell_time': []},
            'high_quality': {'profit': [], 'sell_time': []}
        }
    }

    for q in sorted(results_by_quality.keys()):
        profits = results_by_quality[q]
        times = sell_times_by_quality[q]

        analysis['by_quality'][q] = {
            'count': len(profits),
            'mean_profit': np.mean(profits),
            'std_profit': np.std(profits),
            'mean_sell_time': np.mean(times)
        }

        if q < 0:
            analysis['summary']['low_quality']['profit'].extend(profits)
            analysis['summary']['low_quality']['sell_time'].extend(times)
        elif q == 0:
            analysis['summary']['avg_quality']['profit'].extend(profits)
            analysis['summary']['avg_quality']['sell_time'].extend(times)
        else:
            analysis['summary']['high_quality']['profit'].extend(profits)
            analysis['summary']['high_quality']['sell_time'].extend(times)

    for category in ['low_quality', 'avg_quality', 'high_quality']:
        profits = analysis['summary'][category]['profit']
        times = analysis['summary'][category]['sell_time']

        if profits:
            analysis['summary'][category] = {
                'count': len(profits),
                'mean_profit': np.mean(profits),
                'std_profit': np.std(profits),
                'mean_sell_time': np.mean(times)
            }
        else:
            analysis['summary'][category] = {
                'count': 0,
                'mean_profit': 0,
                'std_profit': 0,
                'mean_sell_time': 0
            }

    return analysis


def print_quality_analysis(analysis):
    """Pretty print quality analysis results."""
    print(f"\n{'='*70}")
    print("POLICY ANALYSIS BY PLAYER QUALITY")
    print(f"{'='*70}")
    print(f"{'Quality':<12} | {'Count':<8} | {'Mean Profit':<15} | {'Std':<12} | {'Sell Time':<10}")
    print("-" * 70)

    for category, label in [('low_quality', 'Low (Q<0)'),
                            ('avg_quality', 'Avg (Q=0)'),
                            ('high_quality', 'High (Q>0)')]:
        stats = analysis['summary'][category]
        print(f"{label:<12} | {stats['count']:<8} | {stats['mean_profit']:>13,.0f} | "
              f"{stats['std_profit']:>10,.0f} | {stats['mean_sell_time']:>8.1f}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("Testing training_utils...\n")

    rewards, q_table = run_training_session(
        env_mode="STOCHASTIC",
        algo="Q_LEARNING",
        num_episodes=10000,
        label="Test Session",
        log_interval=2000
    )

    metrics = evaluate_policy(q_table, num_episodes=500)

    analysis = analyze_policy_by_player_quality(q_table, num_episodes=500)
    print_quality_analysis(analysis)

    converged, conv_ep = detect_convergence(rewards)
    print(f"Convergence detected: {converged}")
    if converged:
        print(f"Convergence episode: {conv_ep}")