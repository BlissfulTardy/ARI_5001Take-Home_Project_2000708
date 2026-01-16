
"""
Implementing the MDP environment for the Manager-Agent simulation.
Handles state transitions, reward calculations, and game mechanics.
"""

import numpy as np
import config as cfg


class FootballEnvironment:
    """
    Football Player Development Environment.
    
    Implements a Finite-Horizon MDP where an agent develops a young
    football player by making sequential training decisions.
    
    State: s = (t, L, K, C)
        - t: Current turn (0 to TURNS_MAX)
        - L: Tuple of 5 attribute levels (0-10 each)
        - K: Tuple of 5 max indicators (0=open, 1=maxed)
        - C: Tuple of 5 potential classifiers
        
    Actions: Train_Def, Train_Pos, Train_Dis, Train_Mob, Train_Sco, Sell
    
    Modes:
        - STOCHASTIC: Max-out determined by Risk Matrix probabilities
        - DETERMINISTIC: Max-out determined by hidden limits
    """
    
    def __init__(self, mode="STOCHASTIC"):
        """
        Initialize the environment.
        
        Args:
            mode: "STOCHASTIC" or "DETERMINISTIC"
        """
        if mode not in ["STOCHASTIC", "DETERMINISTIC"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'STOCHASTIC' or 'DETERMINISTIC'")
        
        self.mode = mode
        self.t = None
        self.L = None
        self.K = None
        self.C = None
        
        # For deterministic mode, store hidden limits
        self._hidden_limits = None
        
        self.reset()
    
    def reset(self):
        """
        Reset environment to initial state with a new random player.
        
        Returns:
            Initial state tuple (t, L, K, C)
        """
        # Time starts at 0
        self.t = 0
        
        # Initialize levels uniformly in [0, 4]
        self.L = np.random.randint(
            cfg.INIT_LEVEL_MIN, 
            cfg.INIT_LEVEL_MAX + 1, 
            size=cfg.NUM_ATTRIBUTES
        )
        
        # All attributes start unmaxed
        self.K = np.zeros(cfg.NUM_ATTRIBUTES, dtype=int)
        
        # Assign potential classifiers based on mode
        if self.mode == "STOCHASTIC":
            self.C = np.random.choice(
                cfg.CLASS_POTS_STO, 
                size=cfg.NUM_ATTRIBUTES, 
                p=cfg.PROB_POTS_STO
            )
            self._hidden_limits = None
        else:  # DETERMINISTIC
            self.C = np.random.choice(
                cfg.CLASS_POTS_DET, 
                size=cfg.NUM_ATTRIBUTES, 
                p=cfg.PROB_POTS_DET
            )
            # Pre-compute hidden limits for each attribute
            self._hidden_limits = np.array([
                cfg.MAP_LIMIT_DET[c] for c in self.C
            ])
        
        return self._get_state_tuple()
    
    def _get_state_tuple(self):
        """
        Convert current state to immutable tuple for hashing.
        
        Returns:
            State tuple (t, tuple(L), tuple(K), tuple(C))
        """
        return (
            self.t,
            tuple(self.L.tolist()),
            tuple(self.K.tolist()),
            tuple(self.C.tolist())
        )
    
    def get_valid_actions(self):
        """
        Get list of valid actions in current state.
        
        Rules:
            - Sell is always valid
            - Train_X is valid only if attribute X is not maxed (K[X] == 0)
            
        Returns:
            List of valid action strings
        """
        valid = ["Sell"]
        
        for i in range(cfg.NUM_ATTRIBUTES):
            if self.K[i] == 0:
                valid.append(cfg.ACTIONS[i])  # Train actions are indices 0-4
        
        return valid
    
    def step(self, action):
        """
        Execute an action and transition to next state.
        
        Implements Algorithm A: EnvironmentStep
        
        Args:
            action: Action string ("Train_Def", "Train_Pos", ..., "Sell")
            
        Returns:
            Tuple (next_state, reward, done)
            - next_state: New state tuple (or None if terminal)
            - reward: Immediate reward (value if Sell, -wage if Train)
            - done: Boolean indicating episode termination
        """
        # --- Terminal Conditions ---
        
        # Forced termination at max turns
        if self.t >= cfg.TURNS_MAX:
            reward = self._compute_player_value()
            return None, reward, True
        
        # Voluntary termination (Sell action)
        if action == "Sell":
            reward = self._compute_player_value()
            return None, reward, True
        
        # --- Training Action ---
        
        # Parse action to get attribute index
        if action not in cfg.TRAIN_ACTION_TO_ATTR_IDX:
            raise ValueError(f"Invalid action: {action}")
        
        attr_idx = cfg.TRAIN_ACTION_TO_ATTR_IDX[action]
        
        # Validate: Cannot train maxed attribute
        if self.K[attr_idx] == 1:
            # This should not happen if agent uses get_valid_actions()
            # Return heavy penalty for invalid action
            return self._get_state_tuple(), -float('inf'), False
        
        # --- Determine Training Outcome ---
        
        success = self._resolve_training(attr_idx)
        
        # --- Execute State Transition ---
        
        if success:
            # Level increases
            self.L[attr_idx] = min(self.L[attr_idx] + 1, cfg.LEVEL_MAX)
        else:
            # Attribute maxes out
            self.K[attr_idx] = 1
        
        # --- Calculate Wage Cost ---
        
        wage_cost = self._compute_wage()
        
        # --- Advance Time ---
        
        self.t += 1
        
        # Check for forced termination
        done = (self.t >= cfg.TURNS_MAX)
        
        # If all attributes are maxed, force sell
        if np.all(self.K == 1):
            # No valid training actions remain
            # Agent will be forced to sell on next step
            pass
        
        return self._get_state_tuple(), -wage_cost, done
    
    def _resolve_training(self, attr_idx):
        """
        Resolve training outcome (success or max-out).
        
        Args:
            attr_idx: Index of attribute being trained
            
        Returns:
            Boolean - True if training successful, False if maxed out
        """
        current_level = self.L[attr_idx]
        potential_class = self.C[attr_idx]
        
        if self.mode == "STOCHASTIC":
            # Look up max-out probability from Risk Matrix
            prob_fail = cfg.RISK_MATRIX[potential_class][current_level]
            
            # Roll for outcome
            roll = np.random.random()
            
            # Success if roll exceeds failure probability
            return roll >= prob_fail
        
        else:  # DETERMINISTIC
            # Check against hidden limit
            limit = self._hidden_limits[attr_idx]
            
            # Success if next level would not exceed limit
            return (current_level + 1) <= limit
    
    def _compute_wage(self):
        """
        Compute wage cost W(s) for current state.
        
        W(s) = (W_base + ψ × ΣL_i) × (1 + φ × t)
        
        Returns:
            Wage cost (float)
        """
        sum_levels = np.sum(self.L)
        base_wage = cfg.WAGE_BASE + cfg.PSI_MATURITY * sum_levels
        inflation = 1.0 + cfg.PHI_INFLATION * self.t
        
        return base_wage * inflation
    
    def _compute_player_value(self):
        """
        Compute player market value V(s).
        
        V(s) = V_now(s) + V_gap(s) × δ(t)
        
        Where:
        - V_now = P_base × β^Υ_now
        - V_gap = P_spec × (β^Υ_future - β^Υ_now)
        - δ(t) = min(1, t/T_peak) × ((T_max - t) / T_max)^λ
        
        Returns:
            Market value (float)
        """
        # --- Compute Realized Proficiency (Υ_now) ---
        
        upsilon_now = self._compute_best_proficiency(self.L)
        
        # --- Compute Expected Levels ---
        
        L_expected = self._compute_expected_levels()
        
        # --- Compute Projected Proficiency (Υ_future) ---
        
        upsilon_future = self._compute_best_proficiency(L_expected)
        
        # --- Compute Value Components ---
        
        v_now = cfg.PRICE_BASE * (cfg.BETA_PROF ** upsilon_now)
        v_future = cfg.PRICE_BASE * (cfg.BETA_PROF ** upsilon_future)
        v_gap = cfg.PRICE_SPECULATIVE * (
            (cfg.BETA_PROF ** upsilon_future) - (cfg.BETA_PROF ** upsilon_now)
        )
        
        # --- Compute Time Decay with Ramp-Up ---
        
        delta = self._compute_delta()
        
        # --- Final Value ---
        
        return v_now + v_gap * delta
    
    def _compute_best_proficiency(self, levels):
        """
        Compute best role proficiency for given levels.
        
        Υ = max over all roles of (Σ w_i × L_i)
        
        Args:
            levels: Array of 5 attribute levels
            
        Returns:
            Best proficiency value (float)
        """
        best_prof = 0.0
        
        for role in cfg.ROLES:
            weights = cfg.WEIGHT_ROLES[role]
            prof = np.dot(levels, weights)
            if prof > best_prof:
                best_prof = prof
        
        return best_prof
    
    def _compute_expected_levels(self):
        """
        Compute expected final levels for each attribute.
        
        For each attribute:
        - If maxed (K[i] == 1): Expected = Current level
        - If open (K[i] == 0): Expected = E[L|C] from Expected Levels Matrix
        
        Returns:
            Array of 5 expected levels
        """
        expected = np.zeros(cfg.NUM_ATTRIBUTES)
        
        for i in range(cfg.NUM_ATTRIBUTES):
            if self.K[i] == 1:
                # Already maxed - level is fixed
                expected[i] = self.L[i]
            else:
                if self.mode == "STOCHASTIC":
                    # Look up from Expected Levels Matrix
                    level = min(self.L[i], cfg.LEVEL_MAX)
                    potential = self.C[i]
                    expected[i] = cfg.EXPECTED_LEVELS[potential][level]
                else:  # DETERMINISTIC
                    # Expected level is the hard limit
                    expected[i] = cfg.EXPECTED_LEVELS_DET[self.C[i]]
        
        return expected
    
    def _compute_delta(self):
        """
        Compute modified time-decay factor with ramp-up.
        
        δ(t) = min(1, t/T_peak) × ((T_max - t) / T_max)^λ
        
        Returns:
            Delta value in [0, 1]
        """
        # Ramp-up factor: linear increase from 0 to 1 over T_peak turns
        if cfg.T_PEAK > 0:
            ramp_up = min(1.0, self.t / cfg.T_PEAK)
        else:
            ramp_up = 1.0
        
        # Decay factor: square root decay over remaining time
        if cfg.TURNS_MAX > 0:
            remaining_ratio = (cfg.TURNS_MAX - self.t) / cfg.TURNS_MAX
            decay = remaining_ratio ** cfg.LAMBDA_DECAY
        else:
            decay = 0.0
        
        return ramp_up * decay
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_state_info(self):
        """
        Get detailed information about current state.
        
        Returns:
            Dictionary with state details
        """
        L_exp = self._compute_expected_levels()
        
        return {
            'turn': self.t,
            'levels': self.L.tolist(),
            'maxed': self.K.tolist(),
            'potentials': self.C.tolist(),
            'expected_levels': L_exp.tolist(),
            'proficiency_now': self._compute_best_proficiency(self.L),
            'proficiency_future': self._compute_best_proficiency(L_exp),
            'current_value': self._compute_player_value(),
            'current_wage': self._compute_wage(),
            'delta': self._compute_delta(),
            'valid_actions': self.get_valid_actions()
        }
    
    def render(self):
        """Print current state to console."""
        info = self.get_state_info()
        
        print(f"\n{'='*50}")
        print(f"Turn {info['turn']}/{cfg.TURNS_MAX} | Mode: {self.mode}")
        print(f"{'='*50}")
        
        print(f"\n{'Attr':<6} | {'Level':<6} | {'Pot':<4} | {'Exp':<6} | {'Status':<8}")
        print("-" * 45)
        
        for i, attr in enumerate(cfg.ATTRIBUTES):
            level = info['levels'][i]
            pot = info['potentials'][i]
            exp = info['expected_levels'][i]
            status = "MAXED" if info['maxed'][i] else "Open"
            
            print(f"{attr:<6} | {level:<6} | {pot:<4} | {exp:<6.2f} | {status:<8}")
        
        print(f"\nΥ_now: {info['proficiency_now']:.2f}")
        print(f"Υ_future: {info['proficiency_future']:.2f}")
        print(f"δ(t): {info['delta']:.3f}")
        print(f"Value: {info['current_value']:,.0f}")
        print(f"Wage: {info['current_wage']:,.0f}")
        print(f"\nValid Actions: {info['valid_actions']}")


class EnvironmentStats:
    """
    Utility class for computing environment statistics.
    """
    
    @staticmethod
    def compute_delta_profile():
        """Compute δ(t) for all turns."""
        deltas = []
        for t in range(cfg.TURNS_MAX + 1):
            if cfg.T_PEAK > 0:
                ramp = min(1.0, t / cfg.T_PEAK)
            else:
                ramp = 1.0
            
            if cfg.TURNS_MAX > 0:
                decay = ((cfg.TURNS_MAX - t) / cfg.TURNS_MAX) ** cfg.LAMBDA_DECAY
            else:
                decay = 0.0
            
            deltas.append(ramp * decay)
        
        return deltas
    
    @staticmethod
    def compute_value_table():
        """Compute value reference table for different proficiency levels."""
        table = {}
        for ups in np.arange(2.0, 10.5, 0.5):
            v_now = cfg.PRICE_BASE * (cfg.BETA_PROF ** ups)
            table[ups] = v_now
        return table
    
    @staticmethod
    def estimate_cumulative_wages(turns):
        """
        Estimate cumulative wages over given turns.
        
        Assumes uniform training (+1 level per turn).
        """
        total = 0.0
        sum_levels = 10  # Average starting sum
        
        for t in range(turns):
            wage = (cfg.WAGE_BASE + cfg.PSI_MATURITY * sum_levels) * (1 + cfg.PHI_INFLATION * t)
            total += wage
            sum_levels += 1  # One level gained per turn
        
        return total


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("Testing FootballEnvironment...\n")
    
    # Test stochastic mode
    env = FootballEnvironment(mode="STOCHASTIC")
    state = env.reset()
    env.render()
    
    print("\n--- Taking some actions ---\n")
    
    for step in range(5):
        valid_actions = env.get_valid_actions()
        # Pick a random training action (not Sell)
        train_actions = [a for a in valid_actions if a != "Sell"]
        
        if train_actions:
            action = np.random.choice(train_actions)
        else:
            action = "Sell"
        
        print(f"Step {step + 1}: Action = {action}")
        next_state, reward, done = env.step(action)
        
        if done:
            print(f"  -> Terminal! Final Reward: {reward:,.0f}")
            break
        else:
            print(f"  -> Wage Cost: {-reward:,.0f}")
    
    if not done:
        env.render()
    
    print("\n--- Delta Profile ---")
    deltas = EnvironmentStats.compute_delta_profile()
    for t in [0, 5, 10, 15, 20, 30, 40]:
        print(f"  δ({t}) = {deltas[t]:.3f}")
    
    print("\n--- Value Table ---")
    vtable = EnvironmentStats.compute_value_table()
    for ups in [2.0, 4.0, 6.0, 8.0, 10.0]:
        print(f"  Υ={ups}: V_now = {vtable[ups]:,.0f}")
    
    print("\n--- Wage Accumulation ---")
    for t in [10, 20, 30, 40]:
        cum_wage = EnvironmentStats.estimate_cumulative_wages(t)
        print(f"  t={t}: Cumulative Wage ≈ {cum_wage:,.0f}")
