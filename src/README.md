# Manager-Agent MDP: Football Player Development

**ARI 5001 - Principles of Artificial Intelligence**  
**Take-Home Project: Reinforcement Learning in Stochastic Gridworlds**

## Overview

This project simulates a football Manager-Agent responsible for developing young players (ages 16-24) through sequential training decisions. The environment is modeled as a 5-dimensional stochastic gridworld where each dimension represents a player attribute.

The agent must:
- Train attributes strategically while managing the risk of "max-outs"
- Decide when to sell the player for maximum market value
- Balance immediate wage costs against long-term value appreciation

## Project Structure

```
project/
├── config.py              # Configuration and hyperparameters
├── environment.py         # MDP environment implementation
├── agent.py               # Q-Learning and SARSA agents
├── training_utils.py      # Training loops and evaluation utilities
├── experiment_a.py        # Experiment A: Q-Learning vs SARSA
├── experiment_b.py        # Experiment B: Hyperparameter sensitivity
├── experiment_c.py        # Experiment C: Stochastic vs Deterministic
├── experiment_d.py        # Experiment D: Temporal discounting
├── visualize_training.py  # Visualization utilities
├── main.py                # Main entry point
└── README.md              # This file
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install numpy matplotlib
```

## Usage

### Set Current Directory

```bash
cd src
```

### Run All Experiments
```bash
python run_all.py --all
```

### Run Specific Experiment
```bash
python run_all.py --experiment A    # Q-Learning vs SARSA
python run_all.py --experiment B    # Hyperparameter sensitivity (both B1 and B2)
python run_all.py --experiment B1   # Learning rate sensitivity only
python run_all.py --experiment B2   # Exploration decay only
python run_all.py --experiment C    # Stochastic vs Deterministic
python run_all.py --experiment D    # Temporal discounting
```

### Quick Test (Reduced Episodes)
```bash
python run_all.py --test
```

### Custom Output Directory
```bash
python run_all.py --all --output my_results
```

### Visualization
```bash
python visualize_training.py              # Step-by-step training visualization
python visualize_training.py --trained    # Visualize trained policy
python visualize_training.py --compare    # Compare Q-Learning vs SARSA
```

## Experiments

### Experiment A: Algorithmic Convergence & Policy Quality
Compares Q-Learning (off-policy) vs SARSA (on-policy) in terms of:
- Convergence behavior
- Final policy quality
- Risk profile and cumulative regret

### Experiment B: Hyperparameter Sensitivity
- **B1**: Learning rate sensitivity (α ∈ {0.01, 0.1, 0.5})
- **B2**: Exploration decay schedules (Fast, Standard, Slow)

### Experiment C: Deterministic vs Stochastic Transitions
Compares agent behavior under:
- Stochastic: Probabilistic max-outs via Risk Matrix
- Deterministic: Hard limits via pre-determined ceiling

### Experiment D: Temporal Discounting
Tests whether "managerial impatience" (γ < 1.0) improves performance:
- γ ∈ {0.90, 0.95, 0.99, 1.0}
- Analyzes interaction with player quality

## MDP Formulation

### State Space
```
s = (t, L, K, C)
- t: Current turn (0-40)
- L: Attribute levels (5-tuple, each 0-10)
- K: Max indicators (5-tuple, each 0 or 1)
- C: Potential classifiers (5-tuple, each L/M/H)
```

### Action Space
```
A = {Train_Def, Train_Pos, Train_Dis, Train_Mob, Train_Sco, Sell}
```

### Reward Structure
- **Training**: -Wage(s) (step cost)
- **Selling**: +Value(s) (terminal reward)

### Value Function
```
V(s) = V_now(s) + V_gap(s) × δ(t)

Where:
- V_now = P_base × β^Υ_now (realized value)
- V_gap = P_spec × (β^Υ_future - β^Υ_now) (speculative gap)
- δ(t) = min(1, t/T_peak) × ((T_max - t)/T_max)^λ (time decay with ramp-up)
```

## Key Parameters

| Parameter | Value  | Description |
|-----------|--------|-------------|
| PRICE_BASE | 200    | Base multiplier for realized value |
| PRICE_SPECULATIVE | 300    | Base multiplier for speculation |
| BETA_PROF | 2.8    | Exponential growth factor |
| T_PEAK | 15     | Turns until speculation unlocks |
| WAGE_BASE | 1000   | Fixed wage cost per turn |
| NUM_EPISODES | 30,000 | Training episodes |

## Expected Outputs

Results are saved to the `results/` directory:
- `Exp_A_*.png`: Learning curves, regret, distributions
- `Exp_B1_*.png`: Learning rate comparison
- `Exp_B2_*.png`: Exploration schedule comparison
- `Exp_C_*.png`: Stochastic vs Deterministic comparison
- `Exp_D_*.png`: Discount factor analysis

## Author
Efecan Okkalioglu (200708)