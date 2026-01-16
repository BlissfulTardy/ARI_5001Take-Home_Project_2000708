"""
experiment_d.py - Experiment D: Temporal Discounting & Manager Impatience

Investigates whether sub-unity discount factors (Î³ < 1.0) can improve
policy performance in a finite-horizon MDP with inherent time-decay.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import config as cfg
from environment import FootballEnvironment
from agent import ManagerAgent
# ADDED: analyze_efficiency import
from training_utils import run_training_session, analyze_efficiency


def run_experiment_D(output_dir="results_1"):
    """
    Run Experiment D: Temporal Discounting & Manager Impatience.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Temporal Discounting & Manager Impatience")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Storage for results
    # ADDED: 'player_classes' list
    metrics = {g: {
        "profits": [],
        "sell_times": [],
        "quality_scores": [],
        "final_proficiencies": [],
        "player_classes": []
    } for g in cfg.EXP_D_GAMMAS}

    trained_agents = {}
    training_histories = {}

    # --- Phase 1: Training ---
    print("\n" + "-" * 70)
    print("Phase 1: Training Agents with Different Discount Factors")
    print("-" * 70)

    for gamma in cfg.EXP_D_GAMMAS:
        label = f"Gamma_{gamma}"
        print(f"\n>>> Training with Î³ = {gamma}")

        rewards, q_table = run_training_session(
            env_mode="STOCHASTIC",
            algo="Q_LEARNING",
            alpha=0.1,
            gamma=gamma,
            decay_lambda=0.005,
            label=label
        )

        training_histories[gamma] = rewards

        # Create agent with trained Q-table for evaluation
        agent = ManagerAgent(algorithm="Q_LEARNING", alpha=0.1, gamma=gamma)
        agent.Q = q_table
        agent.epsilon = 0.0  # Greedy evaluation
        trained_agents[gamma] = agent

    # --- Phase 2: Detailed Evaluation ---
    print("\n" + "-" * 70)
    print("Phase 2: Detailed Policy Evaluation")
    print("-" * 70)

    N_EVAL = 2000  # More episodes for better statistics
    env = FootballEnvironment(mode="STOCHASTIC")

    for gamma, agent in trained_agents.items():
        print(f"\n>>> Evaluating Î³ = {gamma} over {N_EVAL} episodes...")

        for ep in range(N_EVAL):
            state = env.reset()

            # ADDED: Capture Player Class for PE calculation
            # state[3] is tuple of potentials (e.g. ('L', 'H', ...))
            player_class = state[3][0]

            # Calculate Quality Score: Q = #H - #L
            potentials = state[3]
            quality_score = (
                    sum(1 for p in potentials if p == 'H') -
                    sum(1 for p in potentials if p == 'L')
            )

            total_profit = 0
            sell_turn = cfg.TURNS_MAX
            done = False

            while not done:
                valid_actions = env.get_valid_actions()
                action = agent.get_action(state, valid_actions)

                if action == "Sell":
                    sell_turn = state[0]

                next_state, reward, done = env.step(action)
                total_profit += reward
                state = next_state

            # Record metrics
            metrics[gamma]["profits"].append(total_profit)
            metrics[gamma]["sell_times"].append(sell_turn)
            metrics[gamma]["quality_scores"].append(quality_score)
            metrics[gamma]["player_classes"].append(player_class)  # Store class

            # Record final proficiency
            if env.L is not None:
                final_prof = env._compute_best_proficiency(env.L)
                metrics[gamma]["final_proficiencies"].append(final_prof)

    # --- Phase 3: Statistical Analysis ---
    print("\n" + "-" * 70)
    print("Phase 3: Statistical Analysis")
    print("-" * 70)

    summary = {}

    for gamma in cfg.EXP_D_GAMMAS:
        profits = np.array(metrics[gamma]["profits"])
        sell_times = np.array(metrics[gamma]["sell_times"])
        qualities = np.array(metrics[gamma]["quality_scores"])

        # ADDED: Calculate PE Score
        pe_score = analyze_efficiency(
            metrics[gamma]["profits"],
            metrics[gamma]["player_classes"]
        )

        # Overall statistics
        summary[gamma] = {
            'mean_profit': np.mean(profits),
            'std_profit': np.std(profits),
            'median_profit': np.median(profits),
            'mean_sell_time': np.mean(sell_times),
            'std_sell_time': np.std(sell_times),
            'early_exit_rate': np.mean(sell_times < cfg.TURNS_MAX),
            'pe_score': pe_score  # Store in summary
        }

        # Conditional statistics by quality
        for q_label, q_filter in [('low', qualities < 0),
                                  ('avg', qualities == 0),
                                  ('high', qualities > 0)]:
            if np.any(q_filter):
                summary[gamma][f'profit_{q_label}'] = np.mean(profits[q_filter])
                summary[gamma][f'sell_time_{q_label}'] = np.mean(sell_times[q_filter])
                summary[gamma][f'count_{q_label}'] = np.sum(q_filter)
            else:
                summary[gamma][f'profit_{q_label}'] = 0
                summary[gamma][f'sell_time_{q_label}'] = 0
                summary[gamma][f'count_{q_label}'] = 0

        print(f"\nÎ³ = {gamma}:")
        print(f"  Mean Profit:     {summary[gamma]['mean_profit']:>12,.0f}")
        print(f"  PE Score:        {summary[gamma]['pe_score']:>12.1%}")
        print(f"  Mean Sell Time:  {summary[gamma]['mean_sell_time']:>12.1f}")

    # --- Phase 4: Visualization ---
    print("\n" + "-" * 70)
    print("Phase 4: Generating Visualizations")
    print("-" * 70)

    # Plot 1: Average Net Profit by Gamma
    plt.figure(figsize=(10, 6))
    gammas = cfg.EXP_D_GAMMAS
    avg_profits = [summary[g]['mean_profit'] for g in gammas]
    bars = plt.bar([str(g) for g in gammas], avg_profits,
                   color=['red', 'orange', 'yellowgreen', 'green'],
                   alpha=0.7, edgecolor='black')
    plt.title("Experiment D: Average Net Profit by Discount Factor")
    plt.xlabel("Gamma (Î³)")
    plt.ylabel("Average Net Profit")
    for bar, val in zip(bars, avg_profits):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_D_AvgProfit.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_D_AvgProfit.png")

    # Plot 1b: Average Profit Efficiency by Gamma
    plt.figure(figsize=(10, 6))
    pe_scores = [summary[g]['pe_score'] * 100 for g in gammas]
    bars = plt.bar([str(g) for g in gammas], pe_scores,
                   color=['red', 'orange', 'yellowgreen', 'green'],
                   alpha=0.7, edgecolor='black')
    plt.title("Experiment D: Profit Efficiency by Discount Factor")
    plt.xlabel("Gamma (γ)")
    plt.ylabel("Profit Efficiency (%)")
    plt.ylim(0, 100)
    for bar, val in zip(bars, pe_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_D_ProfitEfficiency.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_D_ProfitEfficiency.png")

    # Plot 2: Average Selling Time
    plt.figure(figsize=(10, 6))
    avg_sell_times = [summary[g]['mean_sell_time'] for g in gammas]
    plt.plot([str(g) for g in gammas], avg_sell_times,
             marker='o', markersize=10, linewidth=2, color='red')
    plt.title("Experiment D: Average Selling Time (Impatience Check)")
    plt.xlabel("Gamma (Î³)")
    plt.ylabel("Average Sell Turn (0-40)")
    plt.ylim(0, 45)
    plt.grid(True, alpha=0.3)
    for i, (g, t) in enumerate(zip(gammas, avg_sell_times)):
        plt.annotate(f'{t:.1f}', (i, t), textcoords="offset points",
                     xytext=(0, 10), ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_D_SellTime.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_D_SellTime.png")

    # Plot 3: Profit by Player Quality
    plt.figure(figsize=(12, 6))
    x = np.arange(len(gammas))
    width = 0.25
    low_profits = [summary[g]['profit_low'] for g in gammas]
    avg_profits_q = [summary[g]['profit_avg'] for g in gammas]
    high_profits = [summary[g]['profit_high'] for g in gammas]
    plt.bar(x - width, low_profits, width, label='Low Quality (Q<0)', color='salmon')
    plt.bar(x, avg_profits_q, width, label='Avg Quality (Q=0)', color='skyblue')
    plt.bar(x + width, high_profits, width, label='High Quality (Q>0)', color='lightgreen')
    plt.title("Experiment D: Profit by Player Quality and Discount Factor")
    plt.xlabel("Gamma (Î³)")
    plt.ylabel("Average Net Profit")
    plt.xticks(x, [str(g) for g in gammas])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_D_ProfitByQuality.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_D_ProfitByQuality.png")

    # Plot 4: Sell Time Distribution
    plt.figure(figsize=(14, 8))
    for idx, gamma in enumerate(gammas):
        plt.subplot(2, 2, idx + 1)
        sell_times = metrics[gamma]["sell_times"]
        plt.hist(sell_times, bins=20, edgecolor='black', alpha=0.7,
                 color=['red', 'orange', 'yellowgreen', 'green'][idx])
        plt.title(f"Î³ = {gamma}")
        plt.xlabel("Sell Turn")
        plt.ylabel("Frequency")
        plt.axvline(np.mean(sell_times), color='black', linestyle='--',
                    label=f'Mean: {np.mean(sell_times):.1f}')
        plt.legend()
    plt.suptitle("Experiment D: Sell Time Distributions by Discount Factor")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_D_SellTimeDistributions.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_D_SellTimeDistributions.png")

    # Plot 5: Learning Curves
    plt.figure(figsize=(12, 6))
    colors = ['red', 'orange', 'yellowgreen', 'green']
    for gamma, color in zip(gammas, colors):
        rewards = training_histories[gamma]
        window = 1000
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
            plt.plot(ma, label=f'Î³ = {gamma}', color=color, alpha=0.8)
    plt.title("Experiment D: Learning Curves by Discount Factor")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Profit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Exp_D_LearningCurves.png"), dpi=150)
    plt.close()
    print(f"  Saved: Exp_D_LearningCurves.png")

    # --- Phase 5: Summary Tables ---
    print("\n" + "=" * 70)
    print("EXPERIMENT D: SUMMARY TABLES")
    print("=" * 70)

    # Table 1: Overall Performance
    print("\n--- Overall Performance by Gamma ---")
    print(f"{'Gamma':<8} | {'Mean Profit':<13} | {'PE Score':<10} | {'Std Profit':<12} | {'Mean Sell':<10}")
    print("-" * 70)

    for gamma in gammas:
        s = summary[gamma]
        # FIXED: Access from 's' (summary), NOT 'm' (metrics)
        print(f"{gamma:<8} | {s['mean_profit']:>13,.0f} | {s['pe_score']:>9.1%} | "
              f"{s['std_profit']:>12,.0f} | {s['mean_sell_time']:>10.1f}")

    # Table 2: Interaction Analysis
    print("\n--- Interaction Analysis (Profit by Player Quality) ---")
    print(f"{'Gamma':<8} | {'Low Q (<0)':<15} | {'Avg Q (=0)':<15} | {'High Q (>0)':<15}")
    print("-" * 60)

    for gamma in gammas:
        s = summary[gamma]
        print(f"{gamma:<8} | {s['profit_low']:>13,.0f} | {s['profit_avg']:>13,.0f} | "
              f"{s['profit_high']:>13,.0f}")

    # Table 3: Sell Time by Quality
    print("\n--- Sell Time by Player Quality ---")
    print(f"{'Gamma':<8} | {'Low Q':<12} | {'Avg Q':<12} | {'High Q':<12}")
    print("-" * 50)

    for gamma in gammas:
        s = summary[gamma]
        print(f"{gamma:<8} | {s['sell_time_low']:>10.1f} | {s['sell_time_avg']:>10.1f} | "
              f"{s['sell_time_high']:>10.1f}")

    # --- Hypothesis Evaluation ---
    print("\n" + "=" * 70)
    print("HYPOTHESIS EVALUATION")
    print("=" * 70)

    # Evaluate based on PE Score if preferred, or Profit
    # Here using Profit as original plan, but can discuss PE in report
    best_gamma = max(gammas, key=lambda g: summary[g]['mean_profit'])
    baseline_profit = summary[1.0]['mean_profit']

    print(f"\nBaseline (Î³=1.0) Profit: {baseline_profit:,.0f}")
    print(f"Best Performing Î³: {best_gamma} with profit {summary[best_gamma]['mean_profit']:,.0f}")

    if best_gamma < 1.0:
        improvement = summary[best_gamma]['mean_profit'] - baseline_profit
        print(f"\nâœ“ H1 SUPPORTED: Impatience (Î³={best_gamma}) improved profit by {improvement:,.0f}")
    else:
        print(f"\nâœ— H1 NOT SUPPORTED: Baseline Î³=1.0 performed best")

    # Check H2
    if summary[0.90]['mean_profit'] < summary[0.95]['mean_profit']:
        print(f"âœ“ H2 SUPPORTED: Î³=0.90 shows excessive panic")
    else:
        print(f"âœ— H2 NOT SUPPORTED: Î³=0.90 did not show excessive panic")

    # Check H3
    low_q_benefit = (summary[0.95]['profit_low'] - summary[1.0]['profit_low'])
    high_q_benefit = (summary[0.95]['profit_high'] - summary[1.0]['profit_high'])

    if low_q_benefit > high_q_benefit:
        print(f"âœ“ H3 SUPPORTED: Discounting benefited low-quality players more")
    else:
        print(f"âœ— H3 NOT SUPPORTED: Discounting did not differentially benefit low-quality")

    print("\n" + "=" * 70)

    return {
        'metrics': metrics,
        'summary': summary,
        'training_histories': training_histories,
        'trained_agents': trained_agents
    }


if __name__ == "__main__":
    results = run_experiment_D()