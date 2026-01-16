
import numpy as np
import config as cfg

# ----------------------------------------------------------------------------
# CODE UTILITY FUNCTIONS

def compute_delta(t):
    """
    Compute the modified time-decay factor with ramp-up.

    δ(t) = min(1, t/T_peak) × ((T_max - t) / T_max)^λ
    """
    # Use cfg.TURN_PEAK for the ramp-up calculation
    if cfg.TURN_PEAK > 0:
        ramp_up = min(1.0, t / cfg.TURN_PEAK)
    else:
        ramp_up = 1.0

    # Use cfg.TURNS_MAX and cfg.LAMBDA_DECAY for the decay calculation
    decay = ((cfg.TURNS_MAX - t) / cfg.TURNS_MAX) ** cfg.LAMBDA_DECAY

    return ramp_up * decay


def compute_proficiency(levels, role_weights):
    """
    Compute proficiency for a given role.

    Υ = Σ(w_i × L_i)
    """
    return np.dot(levels, role_weights)


def compute_best_proficiency(levels):
    """
    Compute the best role proficiency (realized proficiency).

    Υ_now = max over all roles of Υ(role)
    """
    best = 0.0
    # Iterate through roles defined in config
    for role in cfg.ROLES:
        # Access the specific weight array from the config dictionary
        role_weights = cfg.WEIGHT_ROLES[role]
        prof = compute_proficiency(levels, role_weights)
        if prof > best:
            best = prof
    return best


def compute_expected_levels(levels, potentials, maxed):
    """
    Compute expected final levels for each attribute.
    """
    expected = []
    # Use len(cfg.ATTRIBUTES) (which is 5) for the loop
    for i in range(len(cfg.ATTRIBUTES)):
        if maxed[i] == 1:
            # Already maxed - level is fixed
            expected.append(levels[i])
        else:
            # Look up expected level from matrix
            # Clamp level to 10 to ensure valid index lookup
            lvl = min(levels[i], 10)
            pot = potentials[i]

            # Retrieve expected value from config dictionary
            # Handle deterministic 'B' (Bust) case if it appears in potentials but not dict
            if pot in cfg.EXPECTED_LEVELS:
                expected.append(cfg.EXPECTED_LEVELS[pot][lvl])
            elif pot == 'B':
                # Fallback for 'Bust' potential if strict Deterministic mode logic is needed here
                # Assuming standard behavior is flat 4.0 or similar low value
                expected.append(4.0)
            else:
                # Fallback if potential class is unknown
                expected.append(levels[i])

    return np.array(expected)


def compute_value(state):
    """
    Compute player market value V(s).

    V(s) = V_now(s) + V_gap(s) × δ(t)

    Where:
    - V_now = P_base × β^Υ_now
    - V_gap = P_spec × (β^Υ_future - β^Υ_now)
    - δ(t) = min(1, t/T_peak) × ((T_max - t) / T_max)^λ
    """
    t, L, K, C = state

    # Realized proficiency
    upsilon_now = compute_best_proficiency(L)

    # Expected levels and projected proficiency
    L_exp = compute_expected_levels(L, C, K)
    upsilon_future = compute_best_proficiency(L_exp)

    # Value components using updated config names
    # P_base -> cfg.PRICE_BASE
    # β -> cfg.BETA_PROF
    v_now = cfg.PRICE_BASE * (cfg.BETA_PROF ** upsilon_now)

    # Gap calculation
    # P_spec -> cfg.PRICE_SPEC
    proficiency_gain = (cfg.BETA_PROF ** upsilon_future) - (cfg.BETA_PROF ** upsilon_now)
    v_gap = cfg.PRICE_SPEC * proficiency_gain

    # Time decay with ramp-up
    delta = compute_delta(t)

    # Total value
    return v_now + (v_gap * delta)


def compute_wage(state):
    """
    Compute player wage cost W(s).

    W(s) = (W_base + ψ × ΣL_i) × (1 + φ × t)
    """
    t, L, K, C = state

    sum_levels = np.sum(L)

    # Wage calculation using config variables
    # W_base -> cfg.WAGE_BASE
    # ψ -> cfg.PSI_MATURITY
    base_wage = cfg.WAGE_BASE + (cfg.PSI_MATURITY * sum_levels)

    # φ -> cfg.PHI_INFLATION
    inflation = 1 + (cfg.PHI_INFLATION * t)

    return base_wage * inflation

def get_reward(state, action, next_state):
    if action == "Sell" or is_terminal(next_state):
        return compute_value(state)  # Terminal reward
    else:
        return -compute_wage(state)  # Step cost (negative)