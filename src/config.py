
"""
Contains all hyperparameters, constants, and matrices for the football
player development simulation.
"""

import numpy as np

# ============================================================================
# [0] LEARNING HYPERPARAMETERS
# ============================================================================

# --- Agent Parameters ---
ALPHA_BASE = 0.1              # Learning Rate (constant)
GAMMA_BASE = 1.0              # Discount Factor (baseline for finite horizon)
EPSILON_START = 1.0           # Initial Exploration Rate
EPSILON_MIN = 0.01            # Exploration Floor
LAMBDA_EPSILON = 0.005        # Epsilon decay rate (standard)

# ============================================================================
# [1] REWARD FUNCTION PARAMETERS
# ============================================================================

# --- Value Function Parameters ---
PRICE_BASE = 200.0            # Base multiplier for realized value
PRICE_SPECULATIVE = 300.0     # Base multiplier for speculative gap
BETA_PROF = 2.8               # Exponential growth factor per proficiency point

# --- Time Decay / Speculation Window Parameters ---
T_PEAK = 15                   # Turns until speculation fully unlocks
LAMBDA_DECAY = 0.5            # Decay shape exponent (square root)

# --- Wage Function Parameters ---
WAGE_BASE = 1000.0            # Fixed operational cost per turn
PSI_MATURITY = 100.0          # Additional cost per attribute level
PHI_INFLATION = 0.05          # Inflation rate per turn (5%)

# ============================================================================
# [2] SIMULATION CONSTANTS
# ============================================================================

TURNS_START = 0
TURNS_MAX = 40

# Number of attributes
NUM_ATTRIBUTES = 5

# Initial level range [0, 4] inclusive
INIT_LEVEL_MIN = 0
INIT_LEVEL_MAX = 4

# Level bounds
LEVEL_MIN = 0
LEVEL_MAX = 10

# ============================================================================
# [3] TRAINING META-PARAMETERS
# ============================================================================

NUM_EPISODES = 30_000         # Total training episodes
EVAL_WINDOW = 1_000           # Window for moving average calculations
CONVERGENCE_THRESHOLD = 0.05  # Stability margin for convergence detection

LOG_INTERVAL = 1_000         # Episodes between progress logs (previews)

# ============================================================================
# [4] EXPERIMENT CONFIGURATIONS
# ============================================================================

# Experiment B1: Learning Rate Sensitivity
EXP_B1_ALPHAS = [0.01, 0.1, 0.5]

# Experiment B2: Exploration Decay Schedules
EXP_B2_DECAYS = {
    'Fast': 0.05,       # ε < 0.1 within ~200 episodes
    'Standard': 0.005,  # ε < 0.1 within ~2000 episodes
    'Slow': 0.0005      # ε < 0.1 within ~20000 episodes
}

# Experiment D: Discount Factor / Impatience
EXP_D_GAMMAS = [0.90, 0.95, 0.99, 1.0]

# ============================================================================
# [5] LABELS AND IDENTIFIERS
# ============================================================================

# Attribute names
ATTRIBUTES = ["Def", "Pos", "Dis", "Mob", "Sco"]

# Action names (indices 0-4 correspond to training each attribute)
ACTIONS = ["Train_Def", "Train_Pos", "Train_Dis", "Train_Mob", "Train_Sco", "Sell"]

# Role names
ROLES = ["Center Def", "Wide Def", "Midfielder", "Winger", "Forward"]

# ============================================================================
# [6] POTENTIAL CLASSIFIERS
# ============================================================================

# Stochastic mode classifiers
CLASS_POTS_STO = ["L", "M", "H"]

# Deterministic mode classifiers (includes Bust)
CLASS_POTS_DET = ["B", "L", "M", "H"]

# ============================================================================
# [7] PROBABILITY DISTRIBUTIONS
# ============================================================================

# Stochastic Mode: Uniform distribution over {L, M, H}
# Expected level per attribute: E[E_i] = (5.48 + 6.24 + 7.26) / 3 ≈ 6.33
PROB_POTS_STO = [1/3, 1/3, 1/3]

# Deterministic Mode: Calibrated distribution over {B, L, M, H}
# Expected level per attribute: E[λ_i] = 0.06(4) + 0.34(6) + 0.50(8) + 0.10(10) ≈ 7.28
PROB_POTS_DET = [0.06, 0.34, 0.50, 0.10]

# Deterministic limit mapping: potential class -> max level
MAP_LIMIT_DET = {'B': 4, 'L': 6, 'M': 8, 'H': 10}

# ============================================================================
# [8] RISK MATRIX (Stochastic Mode)
# ============================================================================

# Probability of max-out given current level and potential class
# Rows: Potential class (L, M, H)
# Columns: Current level (0-10)
# Value: P(max-out | level, potential)

RISK_MATRIX = {
    #       0     1     2     3     4       5       6       7       8       9       10
    'L': [0.00, 0.00, 0.00, 0.00, 0.2074, 0.4310, 0.8583, 0.9844, 1.0000, 1.0000, 1.0000],
    'M': [0.00, 0.00, 0.00, 0.00, 0.0086, 0.0585, 0.2077, 0.4488, 0.7333, 0.9476, 1.0000],
    'H': [0.00, 0.00, 0.00, 0.00, 0.0000, 0.0000, 0.0000, 0.0078, 0.2157, 0.5089, 1.0000]
}

# ============================================================================
# [9] ROLE WEIGHT MATRIX
# ============================================================================

# Contribution of each attribute to role proficiency
# Each row sums to 1.0
# Columns: [Def, Pos, Dis, Mob, Sco]

WEIGHT_ROLES = {
    "Center Def": np.array([0.6, 0.1, 0.1, 0.1, 0.1]),
    "Wide Def":   np.array([0.4, 0.2, 0.2, 0.2, 0.0]),
    "Midfielder": np.array([0.3, 0.3, 0.3, 0.1, 0.0]),
    "Winger":     np.array([0.1, 0.2, 0.2, 0.3, 0.2]),
    "Forward":    np.array([0.0, 0.1, 0.1, 0.2, 0.6])
}

# ============================================================================
# [10] EXPECTED LEVELS MATRIX
# ============================================================================

# Pre-computed expected final level given current level and potential class
# Calculated via backward recursion from Risk Matrix
# E(l, c) = p_fail(l,c) * l + (1 - p_fail(l,c)) * E(l+1, c)

EXPECTED_LEVELS = {
    #       0     1     2     3     4     5     6     7     8     9     10
    'L': [5.48, 5.48, 5.48, 5.48, 5.48, 5.97, 6.62, 7.37, 8.22, 9.10, 10.0],
    'M': [6.24, 6.24, 6.24, 6.24, 6.24, 6.49, 6.99, 7.65, 8.44, 9.25, 10.0],
    'H': [7.26, 7.26, 7.26, 7.26, 7.26, 7.26, 7.52, 8.02, 8.70, 9.40, 10.0],
}

# For deterministic mode, expected level equals the hard limit
EXPECTED_LEVELS_DET = {
    'B': 4,
    'L': 6,
    'M': 8,
    'H': 10
}

# ============================================================================
# [11] DERIVED CONSTANTS (computed at import time)
# ============================================================================

# Action index mapping
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}
IDX_TO_ACTION = {idx: action for idx, action in enumerate(ACTIONS)}

# Attribute index mapping
ATTR_TO_IDX = {attr: idx for idx, attr in enumerate(ATTRIBUTES)}
IDX_TO_ATTR = {idx: attr for idx, attr in enumerate(ATTRIBUTES)}

# Training action to attribute index mapping
TRAIN_ACTION_TO_ATTR_IDX = {
    "Train_Def": 0,
    "Train_Pos": 1,
    "Train_Dis": 2,
    "Train_Mob": 3,
    "Train_Sco": 4
}


# ============================================================================
# [12] UTILITY FUNCTIONS
# ============================================================================

def get_epsilon_decay_schedule(lambda_val):
    """
    Returns a function that computes epsilon for a given episode.
    
    ε_k = 1 / (1 + λ * k)
    
    Args:
        lambda_val: Decay rate parameter
        
    Returns:
        Function that takes episode number and returns epsilon
    """
    def decay_fn(episode):
        eps = 1.0 / (1.0 + lambda_val * episode)
        return max(eps, EPSILON_MIN)
    return decay_fn


def print_config_summary():
    """Prints a summary of key configuration parameters."""
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"\n[Learning Parameters]")
    print(f"  Alpha (LR):     {ALPHA_BASE}")
    print(f"  Gamma (DF):     {GAMMA_BASE}")
    print(f"  Epsilon Start:  {EPSILON_START}")
    print(f"  Epsilon Min:    {EPSILON_MIN}")
    
    print(f"\n[Value Function]")
    print(f"  Price Base:     {PRICE_BASE}")
    print(f"  Price Spec:     {PRICE_SPECULATIVE}")
    print(f"  Beta:           {BETA_PROF}")
    print(f"  T_Peak:         {T_PEAK}")
    print(f"  Lambda Decay:   {LAMBDA_DECAY}")
    
    print(f"\n[Wage Function]")
    print(f"  Wage Base:      {WAGE_BASE}")
    print(f"  Psi (per lvl):  {PSI_MATURITY}")
    print(f"  Phi (inflate):  {PHI_INFLATION}")
    
    print(f"\n[Simulation]")
    print(f"  Max Turns:      {TURNS_MAX}")
    print(f"  Episodes:       {NUM_EPISODES}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
