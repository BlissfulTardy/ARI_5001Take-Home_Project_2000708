
"""
run_all.py - Main Entry Point for Manager-Agent MDP Experiments
Runs all experiments and generates comprehensive results.
"""

import os
import sys
import time
import argparse
import numpy as np

import config as cfg
from experiment_a import run_experiment_A
from experiment_b import run_experiment_B, run_experiment_B1, run_experiment_B2
from experiment_c import run_experiment_C
from experiment_d import run_experiment_D


def print_intro():
    intro = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║          MANAGER-AGENT MDP: FOOTBALL PLAYER DEVELOPMENT          ║
    ║                                                                  ║
    ║         Reinforcement Learning in Stochastic Gridworlds          ║
    ║                                                                  ║
    ║                    ARI 5001 - Take-Home Project                  ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(intro)


def run_all_experiments(output_dir="results"):
    """
    Run all experiments sequentially.

    Args:
        output_dir: Directory to save all results
    """
    print_intro()

    os.makedirs(output_dir, exist_ok=True)
    
    # Print configuration summary
    cfg.print_config_summary()

    print("\n" + "=" * 70)
    print("STARTING ALL EXPERIMENTS")
    print("=" * 70)

    total_start = time.time()
    results = {}
    
    # --- Experiment A ---
    print("\n" + "█" * 70)
    print("█  EXPERIMENT A: Algorithmic Convergence & Policy Quality")
    print("█" * 70)
    exp_a_start = time.time()
    results['A'] = run_experiment_A(output_dir)
    exp_a_time = time.time() - exp_a_start
    print(f"\nExperiment A completed in {exp_a_time:.1f} seconds")
    
    # --- Experiment B ---
    print("\n" + "█" * 70)
    print("█  EXPERIMENT B: Hyperparameter Sensitivity & Stability")
    print("█" * 70)
    exp_b_start = time.time()
    results['B'] = run_experiment_B(output_dir)
    exp_b_time = time.time() - exp_b_start
    print(f"\nExperiment B completed in {exp_b_time:.1f} seconds")
    
    # --- Experiment C ---
    print("\n" + "█" * 70)
    print("█  EXPERIMENT C: Deterministic vs Stochastic Transitions")
    print("█" * 70)
    exp_c_start = time.time()
    results['C'] = run_experiment_C(output_dir)
    exp_c_time = time.time() - exp_c_start
    print(f"\nExperiment C completed in {exp_c_time:.1f} seconds")
    
    # --- Experiment D ---
    print("\n" + "█" * 70)
    print("█  EXPERIMENT D: Temporal Discounting & Manager Impatience")
    print("█" * 70)
    exp_d_start = time.time()
    results['D'] = run_experiment_D(output_dir)
    exp_d_time = time.time() - exp_d_start
    print(f"\nExperiment D completed in {exp_d_time:.1f} seconds")
    
    # --- Final Summary ---
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"\nTotal Runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nTiming Breakdown:")
    print(f"  Experiment A: {exp_a_time:>8.1f}s")
    print(f"  Experiment B: {exp_b_time:>8.1f}s")
    print(f"  Experiment C: {exp_c_time:>8.1f}s")
    print(f"  Experiment D: {exp_d_time:>8.1f}s")
    
    print(f"\nResults saved to: {os.path.abspath(output_dir)}")
    
    # List generated files
    if os.path.exists(output_dir):
        files = sorted(os.listdir(output_dir))
        print(f"\nGenerated files ({len(files)}):")
        for f in files:
            print(f"  - {f}")
    
    return results


def run_single_experiment(experiment, output_dir="results"):
    """
    Run a single experiment.
    
    Args:
        experiment: Experiment identifier ('A', 'B', 'B1', 'B2', 'C', 'D')
        output_dir: Directory to save results
    """
    print_intro()
    os.makedirs(output_dir, exist_ok=True)
    
    start = time.time()
    
    if experiment == 'A':
        result = run_experiment_A(output_dir)
    elif experiment == 'B':
        result = run_experiment_B(output_dir)
    elif experiment == 'B1':
        result = run_experiment_B1(output_dir)
    elif experiment == 'B2':
        result = run_experiment_B2(output_dir)
    elif experiment == 'C':
        result = run_experiment_C(output_dir)
    elif experiment == 'D':
        result = run_experiment_D(output_dir)
    else:
        print(f"Unknown experiment: {experiment}")
        print("Valid options: A, B, B1, B2, C, D")
        return None
    
    elapsed = time.time() - start
    print(f"\nExperiment {experiment} completed in {elapsed:.1f} seconds")
    
    return result


def quick_test(output_dir="results_test"):
    """
    Run a quick test with reduced episodes.
    
    Useful for verifying code works before full runs.
    """
    print_intro()
    print("\n" + "=" * 70)
    print("QUICK TEST MODE (Reduced Episodes)")
    print("=" * 70)
    
    # Temporarily reduce episode count
    original_episodes = cfg.NUM_EPISODES
    cfg.NUM_EPISODES = 5000  # Much faster
    
    os.makedirs(output_dir, exist_ok=True)
    
    start = time.time()
    
    # Run abbreviated versions
    from training_utils import run_training_session, evaluate_policy
    from environment import FootballEnvironment
    
    print("\n--- Testing Environment ---")
    env = FootballEnvironment(mode="STOCHASTIC")
    state = env.reset()
    print(f"Initial state created: t={state[0]}, L={state[1]}")
    env.render()
    
    print("\n--- Testing Training ---")
    rewards, q_table = run_training_session(
        env_mode="STOCHASTIC",
        algo="Q_LEARNING",
        num_episodes=cfg.NUM_EPISODES,
        label="Quick Test",
        log_interval=1000
    )
    
    print("\n--- Testing Evaluation ---")
    metrics = evaluate_policy(q_table, num_episodes=200)
    
    # Restore original
    cfg.NUM_EPISODES = original_episodes
    
    elapsed = time.time() - start
    print(f"\nQuick test completed in {elapsed:.1f} seconds")
    print("All components working correctly!")
    
    return {
        'rewards': rewards,
        'q_table': q_table,
        'metrics': metrics
    }


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Manager-Agent MDP: Football Player Development Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --all                    Run all experiments
  python run_all.py --experiment A           Run only Experiment A
  python run_all.py --experiment B1          Run only Experiment B1
  python run_all.py --test                   Run quick test
  python run_all.py --output my_results      Save to custom directory
        """
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Run all experiments'
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        choices=['A', 'B', 'B1', 'B2', 'C', 'D'],
        help='Run a specific experiment'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run quick test with reduced episodes'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Default to --all if no arguments provided
    if len(sys.argv) == 1:
        args.all = True
    
    if args.test:
        return quick_test(args.output + "_test")
    elif args.experiment:
        return run_single_experiment(args.experiment, args.output)
    elif args.all:
        return run_all_experiments(args.output)
    else:
        parser.print_help()
        return None


if __name__ == "__main__":
    results = main()
