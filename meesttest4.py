import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import random
from TestEnv import HydroElectric_Test

# Set the seed for reproducibility
np.random.seed(7)
random.seed(7)


class QAgent():

    def __init__(self, discount_rate=0.99):

        """
        Params:
        discount_rate = discount rate used for future rewards
        """

        # create environments (train + validate)
        self.env_train = HydroElectric_Test(path_to_test_data="DATA/train.xlsx")
        self.env_val = HydroElectric_Test(path_to_test_data="DATA/validate.xlsx")

        # keep an "active" env pointer exactly like before
        self.env = self.env_train

        self.best_val_reward = -np.inf
        self.best_Qtable = None

        self.prices_1d_train = self.env_train.price_values.flatten()
        self.prices_1d_val = self.env_val.price_values.flatten()
        self.prices_1d = self.prices_1d_train

        # Set the discount rate
        self.discount_rate = discount_rate

        self.actions = np.array([-1.0, 0, 1.0], dtype=np.float32)
        self.action_space = len(self.actions)

        # === NEW DISCRETIZATION SPECS ===
        # 5 states for reservoir volume
        # 2 states: price above/below 24h average
        # 24 states for hour
        # 2 states: cold month yes/no
        self.volume_states = 5
        self.price_states = 2
        self.hour_states = 24
        self.cold_states = 2

        # Bin edges for digitize:
        # number of bins = len(edges) - 1
        self.volume_bins = np.linspace(0, self.env.max_volume, self.volume_states + 1)  # 6 edges -> 5 bins
        self.binary_bins = np.array([-0.5, 0.5, 1.5], dtype=np.float32)  # 3 edges -> 2 bins
        self.hour_bins = np.arange(0.5, 24.5 + 1e-9, 1.0, dtype=np.float32)  # 25 edges -> 24 bins

        # Bins list in the order of the compact state:
        # [volume, price_above_24h_avg, hour, is_cold_month]
        self.bins = [
            self.volume_bins,
            self.binary_bins,
            self.hour_bins,
            self.binary_bins
        ]

    # ------------------------------
    # Active env switcher (NEW)
    # ------------------------------
    def _set_active_env(self, which: str):
        """
        which: "train" or "val"
        """
        if which == "train":
            self.env = self.env_train
            self.prices_1d = self.prices_1d_train
        elif which in ("val", "validate", "validation"):
            self.env = self.env_val
            self.prices_1d = self.prices_1d_val
        else:
            raise ValueError(f"Unknown env selector: {which}")

    # ------------------------------
    # Feature engineering (NEW)
    # ------------------------------
    def price_above_24h_avg(self, idx, window=24):
        """
        Returns 0.0 if current price <= 24h rolling average, else 1.0.
        """
        prices = self.prices_1d
        start = max(0, idx - window)
        hist = prices[start:idx + 1]  # include current
        current = prices[idx]
        avg = float(np.mean(hist)) if len(hist) > 0 else float(current)
        return 1.0 if float(current) > avg else 0.0

    def is_cold_month(self, month):
        """
        Returns 1.0 if cold month, else 0.0.
        Cold months definition: Nov, Dec, Jan, Feb, Mar.
        """
        cold_months = {11, 12, 1, 2, 3}
        return 1.0 if int(month) in cold_months else 0.0

    def make_compact_state(self, obs, idx):
        """
        obs is the env observation (original format).
        Returns compact features: [volume, price_above_24h_avg, hour, is_cold_month]
        """
        volume = float(obs[0])
        hour = float(obs[2])
        month = float(obs[5])
        pab = float(self.price_above_24h_avg(idx))
        cold = float(self.is_cold_month(month))
        return np.array([volume, pab, hour, cold], dtype=np.float32)

    # ------------------------------
    # Discretization
    # ------------------------------
    def discretize_state(self, state):
        digitized_state = []
        for i in range(len(self.bins)):
            raw_idx = np.digitize(state[i], self.bins[i]) - 1
            safe_idx = int(np.clip(raw_idx, 0, len(self.bins[i]) - 2))
            digitized_state.append(safe_idx)
        return digitized_state

    # ------------------------------
    # Plotting
    # ------------------------------
    def visualize_rewards(self):
        plt.figure(figsize=(7.5, 7.5))

        episodes = 10 * (np.arange(len(self.average_rewards)) + 1)

        plt.plot(episodes, self.average_rewards, label="Shaped reward")
        plt.plot(episodes, self.average_rewards_official, label="Official reward")
        plt.axhline(y=-110, color='r', linestyle='-')
        plt.title('Average reward over the past 100 simulations', fontsize=10)
        plt.legend(['Q-learning performance', 'Benchmark'])
        plt.xlabel('Number of simulations', fontsize=10)
        plt.ylabel('Average reward', fontsize=10)
        plt.savefig('average_rewards.png')
        plt.show()

    def visualize_train_val_error(self):
        """
        Plots train vs validation "error" (defined here as -official_reward).
        Uses the same 10-episode cadence as average_rewards_official.
        """
        plt.figure(figsize=(7.5, 7.5))
        episodes = 10 * (np.arange(len(self.train_error)) + 1)

        plt.plot(episodes, self.train_error, label="Train error (-official reward)")
        plt.plot(episodes, self.val_error, label="Validation error (-official reward)")
        plt.title('Train vs Validation error', fontsize=10)
        plt.xlabel('Number of simulations', fontsize=10)
        plt.ylabel('Error', fontsize=10)
        plt.legend()
        plt.savefig('train_val_error.png')
        plt.show()

    # ------------------------------
    # Helpers
    # ------------------------------
    def calculate_energy_in_tank_in_eu(self, volume, price):
        return (volume * self.env.volume_to_MWh * price)

    def calculate_energy_in_tank(self, volume):
        return (volume * self.env.volume_to_MWh)

    # ------------------------------
    # Q-table
    # ------------------------------
    def create_Q_table(self):
        self.Qtable = np.zeros((
            self.volume_states,  # Volume bins = 5
            self.price_states,   # Above/below 24h avg = 2
            self.hour_states,    # Hour bins = 24
            self.cold_states,    # Cold month = 2
            self.action_space    # Actions
        ))

    # ------------------------------
    # Greedy evaluation
    # ------------------------------
    def evaluate_greedy(self, steps_to_plot: int = 48, which_env: str = "val", plot: bool = True):
        """
        Run one greedy episode (epsilon=0) and (optionally) plot the first `steps_to_plot` timesteps.

        which_env: "train" or "val" (default "val")
        plot: if False, no figures will be shown/saved for this call.
        """
        self._set_active_env(which_env)

        obs, _ = self.env.reset()
        done = False

        # reset inventory trackers to match training logic
        _, real_price, *_ = self.env.observation()
        self.energy_in_tank = self.calculate_energy_in_tank(obs[0])
        self.money_in_tank = self.calculate_energy_in_tank_in_eu(obs[0], real_price)

        # logging
        log_price = []
        log_pab = []
        log_action = []
        log_energy = []
        log_money = []

        total_env_reward = 0.0
        t = 0

        idx = self.env.counter
        compact = self.make_compact_state(obs, idx)
        state = self.discretize_state(compact)

        while not done:
            state_tuple = tuple(state)

            # greedy action
            action_index = int(np.argmax(self.Qtable[state_tuple]))
            action = float(self.actions[action_index])

            # observe BEFORE step (for logging / inventory calc)
            volume_before, price_before, *_ = self.env.observation()

            # step env
            next_obs, reward_env, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_env_reward += float(reward_env)

            # observe AFTER step
            volume_after, price_after, *_ = self.env.observation()

            # --- update inventory trackers exactly like training ---
            delta_volume = volume_after - volume_before
            delta_energy = delta_volume * self.env.volume_to_MWh

            if delta_energy > 0:  # pump (buy)
                self.energy_in_tank += delta_energy
                self.money_in_tank += (1 / 0.8) * price_before * delta_energy

            elif delta_energy < 0:  # sell
                energy_sold = -delta_energy

                if self.energy_in_tank > 1e-9:
                    avg_price_tank = self.money_in_tank / self.energy_in_tank
                else:
                    avg_price_tank = 0.0

                cost_removed = energy_sold * avg_price_tank
                self.energy_in_tank -= energy_sold
                self.money_in_tank -= cost_removed

                if self.energy_in_tank <= 1e-9:
                    self.energy_in_tank = 0.0
                    self.money_in_tank = 0.0
            # -----------------------------------------------

            # log first N steps
            if plot and steps_to_plot > 0 and t < steps_to_plot:
                log_price.append(float(price_before))
                log_action.append(float(action))

                log_energy.append(float(self.energy_in_tank))
                log_money.append(float(self.money_in_tank))

                idx_now = self.env.counter
                log_pab.append(float(self.price_above_24h_avg(idx_now)))

            # prepare next state
            idx = self.env.counter
            compact = self.make_compact_state(next_obs, idx)
            state = self.discretize_state(compact)

            t += 1

        # --- plots (first steps_to_plot only) ---
        if plot and steps_to_plot > 0 and len(log_price) > 0:
            x = np.arange(len(log_price))

            pumps = [i for i, a in enumerate(log_action) if a > 0]
            sells = [i for i, a in enumerate(log_action) if a < 0]
            holds = [i for i, a in enumerate(log_action) if a == 0]

            # Plot 1: price + action markers
            plt.figure(figsize=(10, 4))
            plt.plot(x, log_price)
            if pumps: plt.scatter(pumps, [log_price[i] for i in pumps], marker="^")
            if sells: plt.scatter(sells, [log_price[i] for i in sells], marker="v")
            if holds: plt.scatter(holds, [log_price[i] for i in holds], marker="o", s=20)
            plt.title("Greedy policy (first steps) — Price + actions")
            plt.xlabel("timestep")
            plt.ylabel("price")
            plt.show()

            # Plot 2: above/below 24h avg + action markers
            plt.figure(figsize=(10, 4))
            plt.plot(x, log_pab)
            if pumps: plt.scatter(pumps, [log_pab[i] for i in pumps], marker="^")
            if sells: plt.scatter(sells, [log_pab[i] for i in sells], marker="v")
            if holds: plt.scatter(holds, [log_pab[i] for i in holds], marker="o", s=20)
            plt.title("Greedy policy (first steps) — Price above 24h avg (0/1) + actions")
            plt.xlabel("timestep")
            plt.ylabel("above_24h_avg (0..1)")
            plt.ylim(-0.05, 1.05)
            plt.show()

            # Plot 3: energy in tank
            plt.figure(figsize=(10, 4))
            plt.plot(x, log_energy)
            plt.title("Greedy policy (first steps) — Energy in tank")
            plt.xlabel("timestep")
            plt.ylabel("energy (MWh-equivalent)")
            plt.show()

        print(f"Greedy evaluation on {which_env} (env reward):", total_env_reward)
        return total_env_reward

    # ------------------------------
    # Training
    # ------------------------------
    def train(self, simulations, learning_rate, epsilon=0.05, epsilon_decay=1000, adaptive_epsilon=True,
              adapting_learning_rate=False):

        """
        Params:

        simulations = number of episodes of a game to run
        learning_rate = learning rate for the update equation
        epsilon = epsilon value for epsilon-greedy algorithm
        epsilon_decay = number of full episodes (games) over which the epsilon value will decay to its final value
        adaptive_epsilon = boolean that indicates if the epsilon rate will decay over time or not
        adapting_learning_rate = boolean that indicates if the learning rate should be adaptive or not
        """

        # always train on train env
        self._set_active_env("train")

        self.rewards = []
        self.rewards_official = []
        self.average_rewards = []
        self.average_rewards_official = []
        self.money_in_tank = 0.0
        self.energy_in_tank = 0.0

        # NEW: error tracking (train + val) on same 10-episode cadence
        self.train_error = []
        self.val_error = []

        self.create_Q_table()

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.epsilon_start = 1
        self.epsilon_end = 0.05

        self.lr0 = learning_rate
        self.learning_rate = learning_rate

        if adapting_learning_rate:
            self.learning_rate = 1

        for i in range(simulations):

            print(f'Please wait, the algorithm is learning! The current simulation is {i}')

            # TRAIN RESET (train env)
            self._set_active_env("train")
            obs, _ = self.env.reset()
            _, real_price, *_ = self.env.observation()

            idx = self.env.counter  # should be 0 right after reset
            print("this numer should be 0", idx)

            self.cash = 0.0
            self.money_in_tank = self.calculate_energy_in_tank_in_eu(obs[0], real_price)
            self.energy_in_tank = self.calculate_energy_in_tank(obs[0])

            done = False

            compact = self.make_compact_state(obs, idx)
            state = self.discretize_state(compact)

            total_rewards = 0
            total_reward_official = 0

            if adaptive_epsilon:
                self.epsilon = np.interp(i, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

                if i % 500 == 0 and i <= 1500:
                    print(f"The current epsilon rate is {self.epsilon}")

            if i > self.epsilon_decay:
                self.learning_rate = self.lr0 * 0.05
            else:
                self.learning_rate = self.lr0

            while not done:
                state_tuple = tuple(state)

                # Epsilon-greedy action selection
                if np.random.uniform(0, 1) < self.epsilon:
                    action_index = np.random.randint(self.action_space)
                else:
                    action_index = int(np.argmax(self.Qtable[state_tuple]))

                action = float(self.actions[action_index])

                volume_before, price_before, *_ = self.env.observation()
                inv_before = 0.9 * price_before * self.energy_in_tank - self.money_in_tank

                next_obs, reward_official, terminated, truncated, info = self.env.step(action)

                volume_after, price_after, *_ = self.env.observation()

                delta_volume = volume_after - volume_before
                delta_energy = delta_volume * self.env.volume_to_MWh

                if delta_energy > 0:  # pump (buy)
                    self.energy_in_tank += delta_energy
                    self.money_in_tank += (1 / 0.8) * price_before * delta_energy
                    reward = 0.0

                elif delta_energy < 0:  # sell
                    energy_sold = -delta_energy

                    if self.energy_in_tank > 1e-9:
                        avg_price_tank = self.money_in_tank / self.energy_in_tank
                    else:
                        avg_price_tank = 0.0

                    cost_removed = energy_sold * avg_price_tank
                    revenue = 0.9 * price_before * energy_sold

                    reward = revenue - cost_removed

                    self.energy_in_tank -= energy_sold
                    self.money_in_tank -= cost_removed

                    if self.energy_in_tank <= 1e-9:
                        self.energy_in_tank = 0.0
                        self.money_in_tank = 0.0

                else:
                    reward = 0.0

                inv_after = 0.9 * price_after * self.energy_in_tank - self.money_in_tank

                # potential-based shaping
                reward += self.discount_rate * inv_after - inv_before
                reward = reward / 1000

                done = terminated or truncated

                # Discretize next state (compact)
                idx = self.env.counter
                next_compact = self.make_compact_state(next_obs, idx)
                next_state = self.discretize_state(next_compact)
                next_state_tuple = tuple(next_state)

                reward = float(reward)

                if done:
                    Q_target = reward
                else:
                    Q_target = reward + self.discount_rate * np.max(self.Qtable[next_state_tuple])

                current_q = self.Qtable[state_tuple][action_index]
                self.Qtable[state_tuple][action_index] = current_q + self.learning_rate * (Q_target - current_q)

                total_rewards += reward
                total_reward_official += reward_official
                state = next_state

            if adapting_learning_rate:
                self.learning_rate = self.learning_rate / np.sqrt(i + 1)

            self.rewards.append(total_rewards)
            self.rewards_official.append(total_reward_official)

            print(f'Total reward: {np.mean(self.rewards)}')
            print(f'Total reward_official: {np.mean(self.rewards_official)}')

            if i % 1 == 0:
                # training curves (as before)
                self.average_rewards.append(np.mean(self.rewards))
                self.average_rewards_official.append(np.mean(self.rewards_official))

                # NEW: train/val error
                train_err = -float(np.mean(self.rewards_official))

                # validate greedily on validation env (no plots during training)
                val_reward = float(self.evaluate_greedy(steps_to_plot=0, which_env="val", plot=False))
                val_err = -val_reward

                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    self.best_Qtable = self.Qtable.copy()

                self.train_error.append(train_err)
                self.val_error.append(val_err)

                self.rewards = []
                self.rewards_official = []

        print('The simulation is done!')

        if self.best_Qtable is not None:
            self.Qtable = self.best_Qtable
            print("Loaded best Q-table from validation. Best val reward:", self.best_val_reward)

        np.save("BEST_qtable.npy", self.Qtable)
        print("Q-table saved to qtable.npy, Best greedy is", self.best_val_reward)


agent_standard_greedy = QAgent()
agent_standard_greedy.train(
    simulations=50,
    learning_rate=0.03,
    epsilon=1.0,
    epsilon_decay=40,
    adaptive_epsilon=True
)

# Greedy evaluation defaults to validation env now
agent_standard_greedy.evaluate_greedy()
agent_standard_greedy.visualize_rewards()

# NEW: train vs validation error plot
agent_standard_greedy.visualize_train_val_error()
