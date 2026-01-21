import numpy as np
import matplotlib.pyplot as plt
import random
from TestEnv import HydroElectric_Test

# -------------------- Reproducibility --------------------
SEED = 7
np.random.seed(SEED)
random.seed(SEED)


class QAgent:
    """
    Tabular Q-learning for HydroElectric_Test with:
      - state: (volume_bin, rolling_percentile_bin, hour_bin)
      - actions: continuous levels via index mapping (e.g. [-1, -0.5, 0, 0.5, 1])
      - action masking: can't sell if empty / can't pump if full
      - random episode start by "teleporting" env counter/day/hour (no fast-forward)
      - short episodes (episode_len steps)
      - reward shaping: realized profit + small mark-to-market delta
      - reward scaling for stability
      - greedy evaluation with plots
    """

    def __init__(
        self,
        path="DATA/validate.xlsx",
        discount_rate=0.98,
        lr=0.03,
        # discretization
        volume_bins=21,
        price_bins=8,
        hour_bins=None,
        # features / shaping
        percentile_window=48,
        inv_shape_k=0.01,
        reward_scale=100.0,
        # actions
        actions=None,
        seed=7,
    ):
        np.random.seed(seed)
        random.seed(seed)

        self.env = HydroElectric_Test(path_to_test_data=path)

        # Flatten prices for percentile feature (assumes env uses fixed dataset)
        self.prices_1d = self.env.price_values.flatten()
        self.T = len(self.prices_1d)

        self.gamma = float(discount_rate)
        self.lr = float(lr)

        self.percentile_window = int(percentile_window)
        self.inv_shape_k = float(inv_shape_k)
        self.reward_scale = float(reward_scale)

        # Action set (index -> action sent to env.step)
        if actions is None:
            actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.actions = np.array(actions, dtype=np.float32)
        self.n_actions = len(self.actions)

        # Hour bins (edges). 5 regimes (night/morning/noon/evening/late)
        if hour_bins is None:
            hour_bins = np.array([0.5, 6.5, 12.5, 17.5, 21.5, 24.5], dtype=np.float32)
        self.hour_bins = hour_bins

        # Discretization edges (IMPORTANT: edges length = n_bins + 1)
        self.vol_edges = np.linspace(0.0, self.env.max_volume, int(volume_bins) + 1, dtype=np.float32)
        self.pct_edges = np.linspace(0.0, 1.0, int(price_bins) + 1, dtype=np.float32)
        self.hour_edges = self.hour_bins.astype(np.float32)

        self.n_vol = len(self.vol_edges) - 1
        self.n_pct = len(self.pct_edges) - 1
        self.n_hour = len(self.hour_edges) - 1

        # Q-table
        self.Q = np.zeros((self.n_vol, self.n_pct, self.n_hour, self.n_actions), dtype=np.float32)

        # Inventory trackers (energy in tank, money spent)
        self.energy_in_tank = 0.0
        self.money_in_tank = 0.0

        # tank capacity in energy units
        self.max_energy = float(self.env.max_volume * self.env.volume_to_MWh)

    # -------------------- Percentile Feature --------------------

    def rolling_percentile(self, idx: int, window: int = None) -> float:
        """
        Percentile rank in [0,1] of price[idx] relative to last `window` points (inclusive).
        """
        if window is None:
            window = self.percentile_window

        idx = int(np.clip(idx, 0, self.T - 1))
        start = max(0, idx - int(window))
        hist = self.prices_1d[start : idx + 1]
        current = self.prices_1d[idx]
        return float(np.mean(hist <= current))

    # -------------------- Discretization --------------------

    @staticmethod
    def _bin_index(x: float, edges: np.ndarray) -> int:
        # returns 0..(len(edges)-2)
        raw = np.digitize([x], edges)[0] - 1
        return int(np.clip(raw, 0, len(edges) - 2))

    def discretize(self, obs) -> tuple:
        """
        obs is the env observation array:
          [volume, price_feature, hour, day_of_week, day_of_year, month, year]
        we use (volume, percentile, hour)
        """
        v = float(obs[0])
        p = float(obs[1])  # already percentile
        h = float(obs[2])
        return (
            self._bin_index(v, self.vol_edges),
            self._bin_index(p, self.pct_edges),
            self._bin_index(h, self.hour_edges),
        )

    # -------------------- Env Teleport (Random Start) --------------------

    def _teleport_env_to_index(self, start_idx: int):
        """
        Jump env to a random time position without stepping.
        Works if env uses:
          - counter (0..T-1)
          - day (1-indexed)
          - hour (1-indexed)
        If day/hour don't exist, we still set counter and then rely on observation().
        """
        start_idx = int(np.clip(start_idx, 0, self.T - 1))
        self.env.counter = start_idx

        if hasattr(self.env, "day") and hasattr(self.env, "hour"):
            self.env.day = start_idx // 24 + 1
            self.env.hour = start_idx % 24 + 1

    # -------------------- Inventory Helpers --------------------

    def _energy_from_volume(self, volume: float) -> float:
        return float(volume * self.env.volume_to_MWh)

    def _money_value(self, volume: float, price: float) -> float:
        return float(volume * self.env.volume_to_MWh * price)

    def _reset_inventory_trackers_from_env(self):
        vol, price, *_ = self.env.observation()
        self.energy_in_tank = self._energy_from_volume(vol)
        self.money_in_tank = self._money_value(vol, price)

    # -------------------- Action Masking --------------------

    def _allowed_actions_mask(self) -> np.ndarray:
        """
        Disallow:
          - selling (action<0) when empty
          - pumping (action>0) when full
        """
        eps = 1e-9
        allowed = np.ones(self.n_actions, dtype=bool)

        if self.energy_in_tank <= eps:
            allowed[self.actions < 0.0] = False

        if self.energy_in_tank >= (self.max_energy - eps):
            allowed[self.actions > 0.0] = False

        # always allow hold
        if not np.any(allowed):
            allowed[self.actions == 0.0] = True

        return allowed

    def _select_action_index(self, state: tuple, epsilon: float) -> int:
        allowed = self._allowed_actions_mask()

        if np.random.rand() < epsilon:
            return int(np.random.choice(np.where(allowed)[0]))

        q = self.Q[state].copy()
        q[~allowed] = -1e30
        return int(np.argmax(q))

    # -------------------- Reward + Step --------------------

    def _step_and_reward(self, action: float):
        """
        Step env with continuous action.
        Returns: next_obs, shaped_reward, env_reward, done
        """
        # observe BEFORE
        vol_before, price_before, *_ = self.env.observation()
        inv_before = 0.9 * price_before * self.energy_in_tank - self.money_in_tank

        # step
        next_obs, env_reward, terminated, truncated, _ = self.env.step(float(action))
        done = bool(terminated or truncated)

        # observe AFTER
        vol_after, price_after, *_ = self.env.observation()

        # realized profit bookkeeping (your logic)
        delta_vol = vol_after - vol_before
        delta_energy = float(delta_vol * self.env.volume_to_MWh)

        realized = 0.0

        if delta_energy > 0:  # pump (buy)
            self.energy_in_tank += delta_energy
            self.money_in_tank += (1.0 / 0.8) * price_before * delta_energy
            realized = 0.0

        elif delta_energy < 0:  # sell
            energy_sold = -delta_energy
            if self.energy_in_tank > 1e-9:
                avg_price_tank = self.money_in_tank / self.energy_in_tank
            else:
                avg_price_tank = 0.0

            cost_removed = energy_sold * avg_price_tank
            revenue = 0.9 * price_before * energy_sold
            realized = revenue - cost_removed

            self.energy_in_tank -= energy_sold
            self.money_in_tank -= cost_removed

            if self.energy_in_tank <= 1e-9:
                self.energy_in_tank = 0.0
                self.money_in_tank = 0.0

        # shaping: small mark-to-market change
        inv_after = 0.9 * price_after * self.energy_in_tank - self.money_in_tank
        shaped = realized + self.inv_shape_k * (inv_after - inv_before)

        # liquidation at end so it doesn't learn "hold forever"
        if done and self.energy_in_tank > 1e-9:
            avg_price_tank = self.money_in_tank / self.energy_in_tank
            liquidation_profit = 0.9 * self.energy_in_tank * (price_after - avg_price_tank)
            shaped += float(liquidation_profit)
            self.energy_in_tank = 0.0
            self.money_in_tank = 0.0

        # scale reward for stability
        shaped /= self.reward_scale

        return next_obs, float(shaped), float(env_reward), done

    # -------------------- Training --------------------

    def train(
        self,
        episodes=3000,
        episode_len=168,          # 7 days
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_frac=0.9,
        random_start=True,
        log_every=50,
    ):
        """
        Q-learning on short randomized episodes.
        """
        decay_episodes = max(1, int(episodes * epsilon_decay_frac))

        env_returns = []
        shaped_returns = []

        for ep in range(episodes):
            # epsilon schedule
            if ep <= decay_episodes:
                epsilon = float(np.interp(ep, [0, decay_episodes], [epsilon_start, epsilon_end]))
            else:
                epsilon = float(epsilon_end)

            # reset
            obs, _ = self.env.reset()

            # choose a random valid start index (must allow percentile window and episode length)
            if random_start:
                start_min = self.percentile_window
                start_max = self.T - episode_len - 2
                if start_max > start_min:
                    start = np.random.randint(start_min, start_max)
                    self._teleport_env_to_index(start)
                    obs = self.env.observation()

            # sync inventory
            self._reset_inventory_trackers_from_env()

            # set percentile feature for starting obs
            idx = int(getattr(self.env, "counter", 0))
            obs[1] = self.rolling_percentile(idx)

            state = self.discretize(obs)

            total_env = 0.0
            total_shaped = 0.0
            done = False
            t = 0

            while (not done) and (t < episode_len):
                a_idx = self._select_action_index(state, epsilon)
                action = float(self.actions[a_idx])

                next_obs, r_shaped, r_env, done = self._step_and_reward(action)

                # next state's percentile feature
                idx = int(getattr(self.env, "counter", 0))
                next_obs[1] = self.rolling_percentile(idx)
                next_state = self.discretize(next_obs)

                # Q update
                best_next = float(np.max(self.Q[next_state]))
                target = r_shaped + (0.0 if done else self.gamma * best_next)
                self.Q[state + (a_idx,)] += self.lr * (target - self.Q[state + (a_idx,)])

                total_env += r_env
                total_shaped += r_shaped

                state = next_state
                t += 1

            env_returns.append(total_env)
            shaped_returns.append(total_shaped)

            if (ep + 1) % log_every == 0:
                print(
                    f"ep {ep+1}/{episodes} | eps={epsilon:.3f} | "
                    f"env_avg(last {log_every})={np.mean(env_returns[-log_every:]):.1f} | "
                    f"shaped_avg(last {log_every})={np.mean(shaped_returns[-log_every:]):.3f}"
                )

        return env_returns, shaped_returns

    # -------------------- Greedy Evaluation + Plots --------------------

    def evaluate_greedy(
        self,
        episode_len=168,
        steps_to_plot=48,
        random_start=True,
        title_prefix="Greedy",
    ):
        obs, _ = self.env.reset()

        # random start
        if random_start:
            start_min = self.percentile_window
            start_max = self.T - episode_len - 2
            if start_max > start_min:
                start = np.random.randint(start_min, start_max)
                self._teleport_env_to_index(start)
                obs = self.env.observation()

        self._reset_inventory_trackers_from_env()

        idx = int(getattr(self.env, "counter", 0))
        obs[1] = self.rolling_percentile(idx)
        state = self.discretize(obs)

        log_price, log_pct, log_action, log_energy, log_env_r = [], [], [], [], []
        total_env = 0.0
        total_shaped = 0.0

        done = False
        t = 0

        while (not done) and (t < episode_len):
            allowed = self._allowed_actions_mask()
            q = self.Q[state].copy()
            q[~allowed] = -1e30
            a_idx = int(np.argmax(q))
            action = float(self.actions[a_idx])

            # log BEFORE step
            vol_b, price_b, *_ = self.env.observation()
            if t < steps_to_plot:
                log_price.append(float(price_b))
                log_pct.append(float(obs[1]))
                log_action.append(float(action))
                log_energy.append(float(self.energy_in_tank))

            next_obs, r_shaped, r_env, done = self._step_and_reward(action)

            idx = int(getattr(self.env, "counter", 0))
            next_obs[1] = self.rolling_percentile(idx)
            obs = next_obs
            state = self.discretize(obs)

            total_env += r_env
            total_shaped += r_shaped
            if t < steps_to_plot:
                log_env_r.append(float(r_env))

            t += 1

        # ---- plots ----
        x = np.arange(len(log_price))
        pumps = [i for i, a in enumerate(log_action) if a > 0.0]
        sells = [i for i, a in enumerate(log_action) if a < 0.0]
        holds = [i for i, a in enumerate(log_action) if a == 0.0]

        plt.figure(figsize=(10, 4))
        plt.plot(x, log_price)
        if pumps: plt.scatter(pumps, [log_price[i] for i in pumps], marker="^")
        if sells: plt.scatter(sells, [log_price[i] for i in sells], marker="v")
        if holds: plt.scatter(holds, [log_price[i] for i in holds], marker="o", s=20)
        plt.title(f"{title_prefix} — Price + actions (first {steps_to_plot})")
        plt.xlabel("timestep")
        plt.ylabel("price")
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(x, log_pct)
        if pumps: plt.scatter(pumps, [log_pct[i] for i in pumps], marker="^")
        if sells: plt.scatter(sells, [log_pct[i] for i in sells], marker="v")
        if holds: plt.scatter(holds, [log_pct[i] for i in holds], marker="o", s=20)
        plt.title(f"{title_prefix} — Percentile + actions (first {steps_to_plot})")
        plt.xlabel("timestep")
        plt.ylabel("percentile (0..1)")
        plt.ylim(-0.05, 1.05)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(x, log_energy)
        plt.title(f"{title_prefix} — Energy in tank (first {steps_to_plot})")
        plt.xlabel("timestep")
        plt.ylabel("energy (MWh-equivalent)")
        plt.show()

        plt.figure(figsize=(10, 3))
        plt.plot(x, log_env_r)
        plt.title(f"{title_prefix} — Env reward per step (first {steps_to_plot})")
        plt.xlabel("timestep")
        plt.ylabel("env reward")
        plt.show()

        print(f"{title_prefix} eval: env_total={total_env:.1f} | shaped_total={total_shaped:.3f}")
        return total_env, total_shaped


# -------------------- Run --------------------
if __name__ == "__main__":
    agent = QAgent(
        discount_rate=0.98,
        lr=0.03,

        volume_bins=21,
        price_bins=8,
        percentile_window=48,
        inv_shape_k=0.01,       # keep small; try 0.02 if too conservative
        reward_scale=100.0,     # stabilizes learning
        actions=[-1.0, -0.5, 0.0, 0.5, 1.0],
        seed=SEED,
    )

    env_rewards, shaped_rewards = agent.train(
        episodes=3000,
        episode_len=168,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_frac=0.9,
        random_start=True,
        log_every=50,
    )

    agent.evaluate_greedy(
        episode_len=168,
        steps_to_plot=48,
        random_start=True,
        title_prefix="Greedy",
    )

    # Learning curve (env reward moving average)
    window = 50
    if len(env_rewards) >= window:
        ma = np.convolve(env_rewards, np.ones(window) / window, mode="valid")
        plt.figure(figsize=(10, 4))
        plt.plot(ma)
        plt.title(f"Env reward moving average (window={window})")
        plt.xlabel("episode")
        plt.ylabel("env reward")
        plt.show()
