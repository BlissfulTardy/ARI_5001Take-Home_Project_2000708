"""
experiment_c.py - Experiment C: Deterministic vs Stochastic Transitions

Compares agent behavior and policy structure under:
- Stochastic: Probabilistic max-outs via Risk Matrix
- Deterministic: Hard limits via pre-determined ceiling
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import config as cfg
from training_utils import (
    run_training_session,
    evaluate_policy,
    compute_moving_average,
    analyze_policy_by_player_quality,
    print_quality_analysis
)


def run_experiment_C(output_dir="results_1"):
    """
    Run Experiment C: Deterministic vs Stochastic Transitions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Deterministic vs Stochastic Transitions")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # --- Train Stochastic Agent ---
    print("\n[1/2] Training in STOCHASTIC Environment...")
    rewards_sto, q_table_sto = run_training_session(
        env_mode="STOCHASTIC",
        algo="Q_LEARNING",
        alpha=0.1,
        gamma=1.0,
        decay_lambda=0.005,
        label="Stochastic"
    )

    results['Stochastic'] = {
        'rewards': rewards_sto,
        'q_table': q_table_sto
    }

    # --- Train Deterministic Agent ---
    print("\n[2/2] Training in DETERMINISTIC Environment...")
    rewards_det, q_table_det = run_training_session(
        env_mode="DETERMINISTIC",
        algo="Q_LEARNING",
        alpha=0.1,
        gamma=1.0,
        decay_lambda=0.005,
        label="Deterministic"
    )

    results['Deterministic'] = {
        'rewards': rewards_det,
        'q_table': q_table_det
    }

    # --- Analysis (Training Metrics) ---
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    for mode in ['Stochastic', 'Deterministic']:
        rewards = results[mode]['rewards']
        ma = compute_moving_average(rewards, window=cfg.EVAL_WINDOW)
        final_rewards = rewards[-cfg.EVAL_WINDOW:]

        results[mode]['metrics'] = {
            'moving_avg': ma,
            'mean_profit': np.mean(final_rewards),
            'std_profit': np.std(final_rewards),
            'min_profit': np.min(final_rewards),
            'max_profit': np.max(final_rewards)
        }

        print(f"\n--- {mode} Environment ---")
        print(f"  Mean Profit: {results[mode]['metrics']['mean_profit']:>12,.0f}")
        print(f"  Std Profit:  {results[mode]['metrics']['std_profit']:>12,.0f}")

    # --- Policy Evaluation (Getting PE Score) ---
    print("\n" + "=" * 70)
    print("POLICY EVALUATION")
    print("=" * 70)

    for mode in ['Stochastic', 'Deterministic']:
        print(f"\n--- {mode} Policy ---")
        env_mode = "STOCHASTIC" if mode == "Stochastic" else "DETERMINISTIC"
        eval_metrics = evaluate_policy(
            results[mode]['q_table'],
            env_mode=env_mode,
            num_episodes=1000,
            verbose=True
        )
        # Store these separately!
        results[mode]['eval_metrics'] = eval_metrics

    # --- Visualization ---
    print("\nGenerating plots...")

    # Get eval metrics for plots
    st_eval = results['Stochastic']['eval_metrics']
    dt_eval = results['Deterministic']['eval_metrics']

    # Plot 1: Learning Curves
    plt.figure(figsize=(12, 6))
    ma_sto = results['Stochastic']['metrics']['moving_avg']
    ma_det = results['Deterministic']['metrics']['moving_avg']
    plt.plot(ma_sto, label='Stochastic (Risk Matrix)', color='blue', alpha=0.8)
    plt.plot(ma_det, label='Deterministic (Hidden Limits)', color='orange', alpha=0.8)
    plt.title("Experiment C: Stochastic vs Deterministic Learning Curves")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Profit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_C_LearningCurves.png"), dpi=150)
    plt.close()

    # Plot 2: Profit Distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(results['Stochastic']['rewards'][-1000:], bins=30, edgecolor='black', alpha=0.7, color='blue')
    plt.title("Stochastic: Profit Distribution")
    plt.subplot(1, 2, 2)
    plt.hist(results['Deterministic']['rewards'][-1000:], bins=30, edgecolor='black', alpha=0.7, color='orange')
    plt.title("Deterministic: Profit Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_C_ProfitDistributions.png"), dpi=150)
    plt.close()

    # Plot 3: Sell Time
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # Correctly access sell times from eval_metrics
    sell_times_sto = results['Stochastic']['eval_metrics']['sell_times']
    plt.hist(sell_times_sto, bins=20, edgecolor='black', alpha=0.7, color='blue')
    plt.title("Stochastic: Sell Time Distribution")
    plt.subplot(1, 2, 2)
    sell_times_det = results['Deterministic']['eval_metrics']['sell_times']
    plt.hist(sell_times_det, bins=20, edgecolor='black', alpha=0.7, color='orange')
    plt.title("Deterministic: Sell Time Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_C_SellTimeComparison.png"), dpi=150)
    plt.close()

    # Plot 4: Profit Efficiency Comparison
    plt.figure(figsize=(10, 6))
    pe_scores = [st_eval.get('pe_score', 0) * 100, dt_eval.get('pe_score', 0) * 100]
    bars = plt.bar(['Stochastic', 'Deterministic'], pe_scores,
                   color=['blue', 'orange'], alpha=0.7, edgecolor='black')
    plt.title("Experiment C: Profit Efficiency - Stochastic vs Deterministic")
    plt.ylabel("Profit Efficiency (%)")
    plt.ylim(0, 100)
    for bar, val in zip(bars, pe_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_C_ProfitEfficiency.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_C_ProfitEfficiency.png")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("EXPERIMENT C: SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<25} | {'Stochastic':>15} | {'Deterministic':>15}")
    print("-" * 60)

    # 1. Get Training Metrics (for Profit/Std)
    st_train = results['Stochastic']['metrics']
    dt_train = results['Deterministic']['metrics']

    # 2. Get Evaluation Metrics (for PE / Sell Time)
    st_eval = results['Stochastic']['eval_metrics']
    dt_eval = results['Deterministic']['eval_metrics']

    print(f"{'Mean Profit':<25} | {st_train['mean_profit']:>15,.0f} | {dt_train['mean_profit']:>15,.0f}")

    # Access PE Score from EVAL metrics
    print(f"{'Profit Efficiency (PE)':<25} | {st_eval.get('pe_score', 0):>15.1%} | {dt_eval.get('pe_score', 0):>15.1%}")

    print(f"{'Std Profit':<25} | {st_train['std_profit']:>15,.0f} | {dt_train['std_profit']:>15,.0f}")

    # Access Sell Time from EVAL metrics
    print(f"{'Mean Sell Time':<25} | {st_eval['mean_sell_time']:>15.1f} | {dt_eval['mean_sell_time']:>15.1f}")

    print("\n" + "=" * 70)
    print("Note: Deterministic environment has ~0.95 higher expected levels")
    print("due to elimination of stochastic max-out risk ('risk premium').")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_experiment_C()