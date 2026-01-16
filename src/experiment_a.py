
"""
experiment_a.py - Experiment A: Algorithmic Convergence & Policy Quality

Compares Q-Learning (Off-Policy) vs SARSA (On-Policy) control
in terms of convergence behavior, policy quality, and risk profiles.

Metrics now include Profit Efficiency (PE) alongside raw Profit.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import config as cfg
from training_utils import (
    run_training_session,
    evaluate_policy,
    compute_moving_average,
    detect_convergence,
    compute_cumulative_regret,
    analyze_policy_by_player_quality,
    print_quality_analysis,
    analyze_efficiency,
    BENCHMARK_PROFITS
)
from environment import FootballEnvironment
from agent import ManagerAgent


def run_experiment_A(output_dir="results"):
    """
    Run Experiment A: Algorithmic Convergence & Policy Quality.
    
    Hypothesis:
    - Q-Learning will converge to higher peak profit but with higher volatility
    - SARSA will converge to lower average profit but with lower regret
    
    Metrics:
    1. Net Profit (moving average)
    2. Development Success Rate
    3. Strategy Yield Consistency (std of rewards)
    
    Args:
        output_dir: Directory to save results_1
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Algorithmic Convergence & Policy Quality")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Fixed parameters
    alpha = 0.1
    gamma = 1.0
    decay_lambda = 0.005  # Standard decay
    
    results = {}
    
    # --- Train Q-Learning Agent ---
    print("\n[1/2] Training Q-Learning Agent...")
    rewards_ql, q_table_ql = run_training_session(
        env_mode="STOCHASTIC",
        algo="Q_LEARNING",
        alpha=alpha,
        gamma=gamma,
        decay_lambda=decay_lambda,
        label="Q-Learning"
    )
    results['Q-Learning'] = {
        'rewards': rewards_ql,
        'q_table': q_table_ql
    }
    
    # --- Train SARSA Agent ---
    print("\n[2/2] Training SARSA Agent...")
    rewards_sarsa, q_table_sarsa = run_training_session(
        env_mode="STOCHASTIC",
        algo="SARSA",
        alpha=alpha,
        gamma=gamma,
        decay_lambda=decay_lambda,
        label="SARSA"
    )
    results['SARSA'] = {
        'rewards': rewards_sarsa,
        'q_table': q_table_sarsa
    }
    
    # --- Analysis ---
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Compute metrics for both algorithms
    for algo in ['Q-Learning', 'SARSA']:
        rewards = results[algo]['rewards']
        
        # Moving average
        ma = compute_moving_average(rewards, window=cfg.EVAL_WINDOW)
        
        # Convergence detection
        converged, conv_ep = detect_convergence(rewards)
        
        # Final statistics
        final_rewards = rewards[-cfg.EVAL_WINDOW:]
        mean_profit = np.mean(final_rewards)
        std_profit = np.std(final_rewards)
        
        # Cumulative regret
        regret = compute_cumulative_regret(rewards)
        final_regret = regret[-1]
        
        results[algo]['metrics'] = {
            'moving_avg': ma,
            'converged': converged,
            'convergence_episode': conv_ep,
            'mean_profit': mean_profit,
            'std_profit': std_profit,
            'cumulative_regret': regret,
            'final_regret': final_regret
        }
        
        print(f"\n--- {algo} ---")
        print(f"  Converged:          {converged} (Episode {conv_ep})")
        print(f"  Mean Profit:        {mean_profit:>12,.0f}")
        print(f"  Std Profit:         {std_profit:>12,.0f}")
        print(f"  Final Regret:       {final_regret:>12,.0f}")
    
    # --- Evaluate Trained Policies ---
    print("\n" + "=" * 70)
    print("POLICY EVALUATION (Greedy, No Exploration)")
    print("=" * 70)
    
    for algo in ['Q-Learning', 'SARSA']:
        print(f"\n--- {algo} Policy ---")
        eval_metrics = evaluate_policy(
            results[algo]['q_table'],
            num_episodes=1000,
            verbose=True
        )
        results[algo]['eval_metrics'] = eval_metrics
        # Store PE score in metrics dict for summary table
        results[algo]['metrics']['pe_score'] = eval_metrics['pe_score']
    
    # --- Quality Analysis ---
    print("\n" + "=" * 70)
    print("POLICY ANALYSIS BY PLAYER QUALITY")
    print("=" * 70)
    
    for algo in ['Q-Learning', 'SARSA']:
        print(f"\n--- {algo} ---")
        quality_analysis = analyze_policy_by_player_quality(
            results[algo]['q_table'],
            num_episodes=1000
        )
        print_quality_analysis(quality_analysis)
        results[algo]['quality_analysis'] = quality_analysis
    
    # --- Visualization ---
    print("\nGenerating plots...")
    
    # Plot 1: Learning Curves (Moving Average Profit)
    plt.figure(figsize=(12, 6))
    
    for algo, color in [('Q-Learning', 'blue'), ('SARSA', 'orange')]:
        ma = results[algo]['metrics']['moving_avg']
        plt.plot(ma, label=algo, color=color, alpha=0.8)
    
    plt.title("Experiment A: Learning Curves (Moving Average Profit)")
    plt.xlabel("Episode")
    plt.ylabel("Average Net Profit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_A_LearningCurves.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_A_LearningCurves.png")
    
    # Plot 2: Cumulative Regret
    plt.figure(figsize=(12, 6))
    
    for algo, color in [('Q-Learning', 'blue'), ('SARSA', 'orange')]:
        regret = results[algo]['metrics']['cumulative_regret']
        plt.plot(regret, label=algo, color=color, alpha=0.8)
    
    plt.title("Experiment A: Cumulative Regret")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_A_CumulativeRegret.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_A_CumulativeRegret.png")
    
    # Plot 3: Profit Distribution (Box Plot)
    plt.figure(figsize=(10, 6))
    
    profit_data = [
        results['Q-Learning']['rewards'][-1000:],
        results['SARSA']['rewards'][-1000:]
    ]
    
    bp = plt.boxplot(profit_data, labels=['Q-Learning', 'SARSA'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightsalmon')
    
    plt.title("Experiment A: Profit Distribution (Final 1000 Episodes)")
    plt.ylabel("Net Profit")
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_A_ProfitDistribution.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_A_ProfitDistribution.png")
    
    # Plot 4: Sell Time Distribution
    plt.figure(figsize=(12, 5))
    
    for idx, algo in enumerate(['Q-Learning', 'SARSA']):
        plt.subplot(1, 2, idx + 1)
        sell_times = results[algo]['eval_metrics']['sell_times']
        plt.hist(sell_times, bins=20, edgecolor='black', alpha=0.7,
                color='blue' if algo == 'Q-Learning' else 'orange')
        plt.title(f"{algo}: Sell Time Distribution")
        plt.xlabel("Sell Turn")
        plt.ylabel("Frequency")
        plt.axvline(np.mean(sell_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sell_times):.1f}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_A_SellTimeDistribution.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_A_SellTimeDistribution.png")
    
    # Plot 5: Profit Efficiency (PE) Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    pe_scores = [results['Q-Learning']['metrics']['pe_score'], 
                 results['SARSA']['metrics']['pe_score']]
    bars = plt.bar(['Q-Learning', 'SARSA'], [p * 100 for p in pe_scores],
                   color=['lightblue', 'lightsalmon'], edgecolor='black')
    plt.title("Experiment A: Profit Efficiency Comparison")
    plt.ylabel("Profit Efficiency (%)")
    plt.ylim(0, 100)
    for bar, val in zip(bars, pe_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1%}', ha='center', va='bottom', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_A_ProfitEfficiency.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_A_ProfitEfficiency.png")
    # In experiment_a.py (inside run_experiment_A)

    # --- Summary Table ---
    print("\n" + "=" * 70)
    print("EXPERIMENT A: SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<25} | {'Q-Learning':>15} | {'SARSA':>15}")
    print("-" * 60)

    # Updated list of metrics to include PE Score
    metrics_to_compare = [
        ('Mean Profit', 'mean_profit', '{:>15,.0f}'),
        ('Profit Efficiency (PE)', 'pe_score', '{:>15.1%}'),  # <--- NEW ROW
        ('Std Profit', 'std_profit', '{:>15,.0f}'),
        ('Final Regret', 'final_regret', '{:>15,.0f}'),
        ('Convergence Episode', 'convergence_episode', '{:>15}'),
    ]

    for name, key, fmt in metrics_to_compare:
        ql_val = results['Q-Learning']['metrics'].get(key, 'N/A')
        sarsa_val = results['SARSA']['metrics'].get(key, 'N/A')

        # Helper to format value safely
        def safe_fmt(val, f_str):
            if val == 'N/A': return val
            try:
                return f_str.format(val)
            except:
                return str(val)

        print(f"{name:<25} | {safe_fmt(ql_val, fmt)} | {safe_fmt(sarsa_val, fmt)}")

    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_experiment_A()
