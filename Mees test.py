import numpy as np
import matplotlib.pyplot as plt
import random
from TestEnv import HydroElectric_Test

# Reproducibility
np.random.seed(7)
random.seed(7)


class QAgent:
    """
    Tabular Q-learning with:
      - discretized state: [volume_bin, price_percentile_bin, hour_bin]
      - continuous actions mapped from indices (e.g. [-1,-0.5,0,0.5,1])
      - action masking (can't sell if empty / can't pump if full)
      - random episode start via fast-forward with no-op
      - short episodes (e.g. 7 days = 168 steps)
      - small inventory shaping term (optional)
    """

    def __init__(
        self,
        path="DATA/validate.xlsx",
        discount_rate=0.98,
        lr=0.05,
        volume_bins=11,
        price_bins=8,
        hour_bins=None,
        actions=None,
        percentile_window=48,
        inv_shape_k=0.05,
        seed=7,
    ):
        np.random.seed(seed)
        random.seed(seed)

        self.env = HydroElectric_Test(path_to_test_data=path)

        # Flatten prices for rolling percentile feature
        self.prices_1d = self.env.price_values.flatten()
        self.percentile_window = int(percentile_window)

        self.gamma = float(discount_rate)
        self.lr = float(lr)
        self.inv_shape_k = float(inv_shape_k)

        # Action set (index -> continuous action passed to env.step)
        if actions is None:
            actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.actions = np.array(actions, dtype=np.float32)
        self.n_actions = len(self.actions)

        # Hour bins (edges) – 5 bins that roughly match typical daily regimes
        # You can tweak these later if you want.
        if hour_bins is None:
            hour_bins = np.array([0.5, 6.5, 12.5, 17.5, 21.5, 24.5], dtype=np.float32)
        self.hour_bins = hour_bins

        # Discretization bins
        self.bin_size = [int(volume_bins), int(price_bins), len(self.hour_bins) - 1]
        self.bins = [
            np.linspace(0.0, self.env.max_volume, self.bin_size[0] + 1),  # volume edges
            np.linspace(0.0, 1.0, self.bin_size[1] + 1),                  # percentile edges
            self.hour_bins,                                               # hour edges
        ]

        # Q-table: [vol_bin, pct_bin, hour_bin, action]
        self.Q = np.zeros((self.bin_size[0], self.bin_size[1], self.bin_size[2], self.n_actions), dtype=np.float32)

        # Max energy in tank (for masking "full")
        self.max_energy = float(self.env.max_volume * self.env.volume_to_MWh)

        # Inventory trackers (energy in tank, money spent)
        self.energy_in_tank = 0.0
        self.money_in_tank = 0.0

    # -------------------- Features --------------------

    def rolling_percentile(self, idx: int, window: int = None) -> float:
        """
        Percentile rank in [0,1] of price at idx relative to last `window` points (inclusive).
        """
        if window is None:
            window = self.percentile_window
        prices = self.prices_1d

        idx = int(np.clip(idx, 0, len(prices) - 1))
        start = max(0, idx - int(window))
        hist = prices[start : idx + 1]
        current = prices[idx]

        # fraction <= current
        return float(np.mean(hist <= current))

    def discretize_state(self, obs):
        """
        obs is expected like: [volume, price_feature, hour, ...]
        We only use first 3 features: volume, percentile, hour
        Returns (vol_bin, pct_bin, hour_bin)
        """
        v, p, h = float(obs[0]), float(obs[1]), float(obs[2])

        inds = []
        for val, edges in [(v, self.bins[0]), (p, self.bins[1]), (h, self.bins[2])]:
            # digitize -> bin index; clip to valid range [0, n_bins-1]
            raw = np.digitize(val, edges) - 1
            n_bins = len(edges) - 1
            inds.append(int(np.clip(raw, 0, n_bins - 1)))
        return tuple(inds)

    # -------------------- Environment helpers --------------------

    def _calc_energy_from_volume(self, volume):
        return float(volume * self.env.volume_to_MWh)

    def _calc_money_value(self, volume, price):
        return float(volume * self.env.volume_to_MWh * price)

    def _reset_inventory_trackers(self):
        # Sync with environment's starting state
        vol, price, *_ = self.env.observation()
        self.energy_in_tank = self._calc_energy_from_volume(vol)
        self.money_in_tank = self._calc_money_value(vol, price)

    def _fast_forward(self, steps: int):
        """
        Advance the environment by `steps` using a no-op action (0.0).
        This gives you a random start time without needing env internals.
        """
        if steps <= 0:
            return

        no_op = 0.0
        done = False
        for _ in range(steps):
            _, _, terminated, truncated, _ = self.env.step(no_op)
            done = terminated or truncated
            if done:
                break

        # If we hit the end of the dataset while fast-forwarding, reset and stop.
        if done:
            self.env.reset()

    # -------------------- Action masking --------------------

    def _allowed_action_indices(self):
        """
        Forbid selling when empty; forbid pumping when full.
        """
        eps = 1e-6
        allowed = np.ones(self.n_actions, dtype=bool)

        # empty => disallow negative actions
        if self.energy_in_tank <= eps:
            allowed[self.actions < 0.0] = False

        # full => disallow positive actions
        if self.energy_in_tank >= (self.max_energy - eps):
            allowed[self.actions > 0.0] = False

        # Always allow hold
        if not np.any(allowed):
            allowed[self.actions == 0.0] = True

        return allowed

    def _select_action_index(self, state, epsilon: float):
        allowed = self._allowed_action_indices()

        if np.random.rand() < epsilon:
            choices = np.where(allowed)[0]
            return int(np.random.choice(choices))

        # greedy over allowed
        qvals = self.Q[state].copy()
        qvals[~allowed] = -1e30
        return int(np.argmax(qvals))

    # -------------------- Reward (realized + shaping) --------------------

    def _step_and_compute_reward(self, action: float):
        """
        Steps env with action, updates inventory trackers like your training code,
        returns: next_obs, shaped_reward, env_reward, done
        """
        # observe BEFORE
        vol_before, price_before, *_ = self.env.observation()
        inv_before = 0.9 * price_before * self.energy_in_tank - self.money_in_tank

        # step env
        next_obs, env_reward, terminated, truncated, info = self.env.step(float(action))
        done = terminated or truncated

        # observe AFTER
        vol_after, price_after, *_ = self.env.observation()

        # inventory delta
        delta_vol = vol_after - vol_before
        delta_energy = float(delta_vol * self.env.volume_to_MWh)

        realized = 0.0

        if delta_energy > 0:  # pumped (bought)
            self.energy_in_tank += delta_energy
            self.money_in_tank += (1.0 / 0.8) * price_before * delta_energy
            realized = 0.0

        elif delta_energy < 0:  # sold
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

        # inventory mark-to-market change (small)
        inv_after = 0.9 * price_after * self.energy_in_tank - self.money_in_tank
        shaped = float(realized + self.inv_shape_k * (inv_after - inv_before))

        # liquidation at end to avoid weird learning about holding forever
        if done and self.energy_in_tank > 1e-9:
            avg_price_tank = self.money_in_tank / self.energy_in_tank
            liquidation_profit = 0.9 * self.energy_in_tank * (price_after - avg_price_tank)
            shaped += float(liquidation_profit)
            self.energy_in_tank = 0.0
            self.money_in_tank = 0.0

        return next_obs, shaped, float(env_reward), done

    # -------------------- Training --------------------

    def train(
        self,
        episodes=3000,
        episode_len=168,         # 7 days
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_frac=0.8,  # decay over 80% of training
        random_start=True,
        max_start_skip=None,     # if None, uses "len(prices)-episode_len-1"
        log_every=50,
    ):
        """
        Q-learning on short randomized episodes.
        """
        decay_episodes = max(1, int(episodes * epsilon_decay_frac))

        rewards_env = []
        rewards_shaped = []

        # For random start: we fast-forward from beginning
        if max_start_skip is None:
            max_start_skip = max(0, len(self.prices_1d) - episode_len - 2)

        for ep in range(episodes):
            # epsilon schedule
            if ep <= decay_episodes:
                epsilon = np.interp(ep, [0, decay_episodes], [epsilon_start, epsilon_end])
            else:
                epsilon = epsilon_end

            obs, _ = self.env.reset()

            # random start by fast-forwarding
            if random_start and max_start_skip > 0:
                skip = random.randint(0, max_start_skip)
                self._fast_forward(skip)

            # sync inventory with env at start point
            self._reset_inventory_trackers()

            # set percentile feature for start state
            idx = int(getattr(self.env, "counter", 0))
            obs[1] = self.rolling_percentile(idx, window=self.percentile_window)

            state = self.discretize_state(obs)

            total_env = 0.0
            total_shaped = 0.0

            done = False
            t = 0

            while (not done) and (t < episode_len):
                a_idx = self._select_action_index(state, epsilon)
                action = float(self.actions[a_idx])

                next_obs, r_shaped, r_env, done = self._step_and_compute_reward(action)

                # next state feature
                idx = int(getattr(self.env, "counter", 0))
                next_obs[1] = self.rolling_percentile(idx, window=self.percentile_window)
                next_state = self.discretize_state(next_obs)

                # Q update (only if not terminal)
                best_next = float(np.max(self.Q[next_state]))
                target = r_shaped + (0.0 if done else self.gamma * best_next)

                self.Q[state + (a_idx,)] += self.lr * (target - self.Q[state + (a_idx,)])

                total_env += r_env
                total_shaped += r_shaped

                state = next_state
                t += 1

            rewards_env.append(total_env)
            rewards_shaped.append(total_shaped)

            if (ep + 1) % log_every == 0:
                print(
                    f"ep {ep+1}/{episodes} | eps={epsilon:.3f} | "
                    f"env_avg(last {log_every})={np.mean(rewards_env[-log_every:]):.1f} | "
                    f"shaped_avg(last {log_every})={np.mean(rewards_shaped[-log_every:]):.1f}"
                )

        return rewards_env, rewards_shaped

    # -------------------- Greedy evaluation + plots --------------------

    def evaluate_greedy(
        self,
        episode_len=168,
        steps_to_plot=48,
        random_start=True,
        max_start_skip=None,
    ):
        if max_start_skip is None:
            max_start_skip = max(0, len(self.prices_1d) - episode_len - 2)

        obs, _ = self.env.reset()

        if random_start and max_start_skip > 0:
            skip = random.randint(0, max_start_skip)
            self._fast_forward(skip)

        self._reset_inventory_trackers()

        idx = int(getattr(self.env, "counter", 0))
        obs[1] = self.rolling_percentile(idx, window=self.percentile_window)
        state = self.discretize_state(obs)

        log_price, log_pct, log_action, log_energy = [], [], [], []
        total_env = 0.0
        total_shaped = 0.0

        done = False
        t = 0

        while (not done) and (t < episode_len):
            # greedy + masking
            allowed = self._allowed_action_indices()
            qvals = self.Q[state].copy()
            qvals[~allowed] = -1e30
            a_idx = int(np.argmax(qvals))
            action = float(self.actions[a_idx])

            # log BEFORE step
            vol_b, price_b, *_ = self.env.observation()
            if t < steps_to_plot:
                log_price.append(float(price_b))
                log_pct.append(float(obs[1]))
                log_action.append(float(action))
                log_energy.append(float(self.energy_in_tank))

            next_obs, r_shaped, r_env, done = self._step_and_compute_reward(action)

            idx = int(getattr(self.env, "counter", 0))
            next_obs[1] = self.rolling_percentile(idx, window=self.percentile_window)
            obs = next_obs
            state = self.discretize_state(obs)

            total_env += r_env
            total_shaped += r_shaped
            t += 1

        # --- plots ---
        x = np.arange(len(log_price))

        pumps = [i for i, a in enumerate(log_action) if a > 0.0]
        sells = [i for i, a in enumerate(log_action) if a < 0.0]
        holds = [i for i, a in enumerate(log_action) if a == 0.0]

        plt.figure(figsize=(10, 4))
        plt.plot(x, log_price)
        if pumps: plt.scatter(pumps, [log_price[i] for i in pumps], marker="^")
        if sells: plt.scatter(sells, [log_price[i] for i in sells], marker="v")
        if holds: plt.scatter(holds, [log_price[i] for i in holds], marker="o", s=20)
        plt.title("Greedy (first steps) — Price + actions")
        plt.xlabel("timestep")
        plt.ylabel("price")
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(x, log_pct)
        if pumps: plt.scatter(pumps, [log_pct[i] for i in pumps], marker="^")
        if sells: plt.scatter(sells, [log_pct[i] for i in sells], marker="v")
        if holds: plt.scatter(holds, [log_pct[i] for i in holds], marker="o", s=20)
        plt.title("Greedy (first steps) — Percentile + actions")
        plt.xlabel("timestep")
        plt.ylabel("percentile (0..1)")
        plt.ylim(-0.05, 1.05)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(x, log_energy)
        plt.title("Greedy (first steps) — Energy in tank")
        plt.xlabel("timestep")
        plt.ylabel("energy (MWh-equivalent)")
        plt.show()

        print(f"Greedy eval: env_reward={total_env:.1f} | shaped_total={total_shaped:.1f}")
        return total_env, total_shaped


# -------------------- Run --------------------

agent = QAgent(
    discount_rate=0.98,
    lr=0.05,
    volume_bins=11,            # more resolution so it can learn "fill vs half vs empty"
    price_bins=8,
    percentile_window=48,
    inv_shape_k=0.05,          # keep this SMALL; tune later (0.02..0.10)
    actions=[-1.0, -0.5, 0.0, 0.5, 1.0],
)

env_rewards, shaped_rewards = agent.train(
    episodes=3000,             # start here; increase if needed
    episode_len=168,           # 7 days
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_frac=0.8,
    random_start=True,
    log_every=50,
)

agent.evaluate_greedy(
    episode_len=168,
    steps_to_plot=48,
    random_start=True,
)

# Optional: learning curve
plt.figure(figsize=(10, 4))
plt.plot(np.convolve(env_rewards, np.ones(50)/50, mode="valid"))
plt.title("Env reward (moving avg, window=50 episodes)")
plt.xlabel("episode")
plt.ylabel("env reward")
plt.show()
