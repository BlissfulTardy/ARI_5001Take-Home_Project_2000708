"""
experiment_b.py - Experiment B: Hyperparameter Sensitivity & Stability

Evaluates how Learning Rate (Î±) and Exploration Schedule (Îµ) impact
the stability and quality of learned value functions.

Sub-experiments:
- B1: Learning Rate Sensitivity (Î± âˆˆ {0.01, 0.1, 0.5})
- B2: Exploration Decay Schedules (Fast, Standard, Slow)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import config as cfg
from training_utils import (
    run_training_session,
    evaluate_policy,
    compute_moving_average,
    detect_convergence
)


def run_experiment_B1(output_dir="results_1"):
    """
    Sub-Experiment B1: Learning Rate Sensitivity.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B1: Learning Rate Sensitivity")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for alpha in cfg.EXP_B1_ALPHAS:
        label = f"Alpha_{alpha}"
        print(f"\n--- Training with Î± = {alpha} ---")

        rewards, q_table = run_training_session(
            env_mode="STOCHASTIC",
            algo="Q_LEARNING",
            alpha=alpha,
            gamma=1.0,
            decay_lambda=0.005,
            label=label
        )

        # 1. Compute Metrics
        ma = compute_moving_average(rewards, window=cfg.EVAL_WINDOW)
        converged, conv_ep = detect_convergence(rewards)
        final_rewards = rewards[-cfg.EVAL_WINDOW:]

        # 2. Run Evaluation to get PE Score
        eval_metrics = evaluate_policy(q_table, num_episodes=1000, verbose=False)

        # Store results (Flat structure)
        results[alpha] = {
            'rewards': rewards,
            'q_table': q_table,
            'moving_avg': ma,
            'converged': converged,
            'convergence_episode': conv_ep,
            'mean_profit': np.mean(final_rewards),
            'std_profit': np.std(final_rewards),
            'q_table_size': len(q_table),
            'pe_score': eval_metrics['pe_score']  # <--- Stored directly here
        }

    # --- Visualization ---
    print("\nGenerating plots...")

    # Plot 1: Learning Curves
    plt.figure(figsize=(12, 6))
    colors = ['green', 'blue', 'red']

    for (alpha, color) in zip(cfg.EXP_B1_ALPHAS, colors):
        ma = results[alpha]['moving_avg']
        plt.plot(ma, label=f'Î± = {alpha}', color=color, alpha=0.8)

    plt.title("Experiment B1: Learning Rate Sensitivity")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Profit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_B1_LearningRates.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_B1_LearningRates.png")

    # Plot 2: Variance Comparison (Rolling Std)
    plt.figure(figsize=(12, 6))

    for (alpha, color) in zip(cfg.EXP_B1_ALPHAS, colors):
        rewards = results[alpha]['rewards']
        window = 1000
        rolling_std = []
        for i in range(window, len(rewards)):
            rolling_std.append(np.std(rewards[i - window:i]))
        plt.plot(rolling_std, label=f'Î± = {alpha}', color=color, alpha=0.8)

    plt.title("Experiment B1: Value Function Stability (Rolling Std)")
    plt.xlabel("Episode")
    plt.ylabel("Rolling Standard Deviation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_B1_Stability.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_B1_Stability.png")

    # Plot 3: Profit Efficiency by Learning Rate
    plt.figure(figsize=(10, 6))
    pe_scores = [results[a]['pe_score'] * 100 for a in cfg.EXP_B1_ALPHAS]
    bars = plt.bar([str(a) for a in cfg.EXP_B1_ALPHAS], pe_scores,
                   color=['green', 'blue', 'red'], alpha=0.7, edgecolor='black')
    plt.title("Experiment B1: Profit Efficiency by Learning Rate")
    plt.xlabel("Learning Rate (α)")
    plt.ylabel("Profit Efficiency (%)")
    plt.ylim(0, 100)
    for bar, val in zip(bars, pe_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_B1_ProfitEfficiency.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_B1_ProfitEfficiency.png")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("EXPERIMENT B1: SUMMARY")
    print("=" * 70)
    print(f"\n{'Alpha':<10} | {'Conv. Ep':<12} | {'Mean Profit':<15} | {'PE Score':<10} | {'Q-States':<10}")
    print("-" * 70)

    for alpha in cfg.EXP_B1_ALPHAS:
        r = results[alpha]
        conv_str = str(r['convergence_episode']) if r['convergence_episode'] else 'N/A'

        # FIXED: Access keys directly from 'r'
        print(f"{alpha:<10} | {conv_str:<12} | {r['mean_profit']:>13,.0f} | "
              f"{r['pe_score']:>9.1%} | {r['q_table_size']:>8,}")

    print("=" * 70)

    return results


def run_experiment_B2(output_dir="results_1"):
    """
    Sub-Experiment B2: Exploration Decay Schedules.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B2: Exploration Decay Schedules")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for decay_name, decay_lambda in cfg.EXP_B2_DECAYS.items():
        label = f"Decay_{decay_name}"
        print(f"\n--- Training with {decay_name} Decay (Î» = {decay_lambda}) ---")

        rewards, q_table = run_training_session(
            env_mode="STOCHASTIC",
            algo="Q_LEARNING",
            alpha=0.1,
            gamma=1.0,
            decay_lambda=decay_lambda,
            label=label
        )

        # 1. Compute Metrics
        ma = compute_moving_average(rewards, window=cfg.EVAL_WINDOW)
        converged, conv_ep = detect_convergence(rewards)
        final_rewards = rewards[-cfg.EVAL_WINDOW:]

        optimal_estimate = np.mean(sorted(rewards)[-100:])
        regret = np.cumsum(optimal_estimate - np.array(rewards))

        # 2. Run Evaluation to get PE Score
        eval_metrics = evaluate_policy(q_table, num_episodes=1000, verbose=False)

        results[decay_name] = {
            'lambda': decay_lambda,
            'rewards': rewards,
            'q_table': q_table,
            'moving_avg': ma,
            'converged': converged,
            'convergence_episode': conv_ep,
            'mean_profit': np.mean(final_rewards),
            'std_profit': np.std(final_rewards),
            'cumulative_regret': regret,
            'final_regret': regret[-1],
            'pe_score': eval_metrics['pe_score']  # <--- Stored directly here
        }

    # --- Visualization ---
    print("\nGenerating plots...")

    # Plot 1: Learning Curves
    plt.figure(figsize=(12, 6))
    colors = {'Fast': 'red', 'Standard': 'blue', 'Slow': 'green'}

    for decay_name in cfg.EXP_B2_DECAYS.keys():
        ma = results[decay_name]['moving_avg']
        plt.plot(ma, label=f'{decay_name} (Î»={results[decay_name]["lambda"]})',
                 color=colors[decay_name], alpha=0.8)

    plt.title("Experiment B2: Exploration Decay Schedules")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Profit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_B2_DecaySchedules.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_B2_DecaySchedules.png")

    # Plot 2: Epsilon Decay Profiles
    plt.figure(figsize=(12, 6))
    episodes = np.arange(1, cfg.NUM_EPISODES + 1)

    for decay_name, decay_lambda in cfg.EXP_B2_DECAYS.items():
        epsilon = 1.0 / (1.0 + decay_lambda * episodes)
        epsilon = np.maximum(epsilon, cfg.EPSILON_MIN)
        plt.plot(episodes, epsilon, label=f'{decay_name} (Î»={decay_lambda})',
                 color=colors[decay_name], alpha=0.8)

    plt.title("Experiment B2: Epsilon Decay Profiles")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon (Îµ)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Îµ = 0.1')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_B2_EpsilonProfiles.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_B2_EpsilonProfiles.png")

    # Plot 3: Cumulative Regret
    plt.figure(figsize=(12, 6))

    for decay_name in cfg.EXP_B2_DECAYS.keys():
        regret = results[decay_name]['cumulative_regret']
        plt.plot(regret, label=decay_name, color=colors[decay_name], alpha=0.8)

    plt.title("Experiment B2: Cumulative Regret by Decay Schedule")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_B2_CumulativeRegret.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_B2_CumulativeRegret.png")

    # Plot 4: Profit Efficiency by Decay Schedule
    plt.figure(figsize=(10, 6))
    decay_names = list(cfg.EXP_B2_DECAYS.keys())
    pe_scores = [results[d]['pe_score'] * 100 for d in decay_names]
    color_list = ['red', 'blue', 'green']
    bars = plt.bar(decay_names, pe_scores, color=color_list, alpha=0.7, edgecolor='black')
    plt.title("Experiment B2: Profit Efficiency by Exploration Decay")
    plt.xlabel("Decay Schedule")
    plt.ylabel("Profit Efficiency (%)")
    plt.ylim(0, 100)
    for bar, val in zip(bars, pe_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_B2_ProfitEfficiency.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_B2_ProfitEfficiency.png")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("EXPERIMENT B2: SUMMARY")
    print("=" * 70)
    print(f"\n{'Decay':<10} | {'Lambda':<10} | {'Mean Profit':<15} | {'PE Score':<10} | {'Final Regret':<15}")
    print("-" * 75)

    for decay_name in cfg.EXP_B2_DECAYS.keys():
        r = results[decay_name]
        conv_str = str(r['convergence_episode']) if r['convergence_episode'] else 'N/A'

        # FIXED: Access keys directly from 'r'
        print(f"{decay_name:<10} | {r['lambda']:<10} | "
              f"{r['mean_profit']:>13,.0f} | {r['pe_score']:>9.1%} | {r['final_regret']:>13,.0f}")

    print("=" * 75)

    return results


def run_experiment_B(output_dir="results_1"):
    """
    Run complete Experiment B (both sub-experiments).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Hyperparameter Sensitivity & Stability")
    print("=" * 70)

    results = {}

    # Run B1: Learning Rates
    results['B1'] = run_experiment_B1(output_dir)

    # Run B2: Exploration Schedules
    results['B2'] = run_experiment_B2(output_dir)

    print("\n" + "=" * 70)
    print("EXPERIMENT B: COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_experiment_B()