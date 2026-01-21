import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import time
import random
from TestEnv import HydroElectric_Test

#Set the seed for reproducibility
np.random.seed(7)
random.seed(7)

class QAgent():

    def __init__(self, discount_rate=0.99, bin_size=20):

        '''
        Params:

        env_name = name of the specific environment that the agent wants to solve
        discount_rate = discount rate used for future rewards
        bin_size = number of bins used for discretizing the state space

        '''

        # create an environment
        self.env = HydroElectric_Test(path_to_test_data = "DATA/validate.xlsx")

        self.prices_1d = self.env.price_values.flatten()

        # Set the discount rate
        self.discount_rate = discount_rate

        self.actions = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        self.action_space = len(self.actions)

        # Set the bin size
        self.bin_size = bin_size

        # State incorporates the observation state
        # Get the low and high values of the environment space
        self.low = np.array([
            0,  # volume
            np.min(self.env.price_values),  # price
            1,  # hour
            0,  # day_of_week
            1,  # day_of_year
            1,  # month
            self.env.timestamps.dt.year.min()  # year
        ], dtype=np.float32)

        self.high = np.array([
            self.env.max_volume,
            np.max(self.env.price_values),
            24,
            6,
            365,
            12,
            self.env.timestamps.dt.year.max()
        ], dtype=np.float32)

        self.bin_size = [5, 8, 5]                #volume = 10, price, hour, day
        self.day_type_bins = np.array([-0.5, 0.5, 1.5])
        self.hour_bins = np.array([0.5, 6.5, 12.5, 17.5, 21.5, 24.5])

        # Manually define meaningful bins
        self.bins = [
            np.linspace(0, self.env.max_volume, self.bin_size[0]),  # Volume
            np.linspace(0, 1, self.bin_size[1]),  # precentile price
            self.hour_bins,
#            self.day_type_bins  # Day of week
        ]

    def discretize_state(self, state):
        digitized_state = []
        for i in range(len(self.bins)):
            raw_idx = np.digitize(state[i], self.bins[i]) - 1
            safe_idx = int(np.clip(raw_idx, 0, len(self.bins[i]) - 2))
            digitized_state.append(safe_idx)
        return digitized_state

    def rolling_percentile(self, idx, window=24):
        '''
        Percentile rank of price.
        Returns a float in [0, 1].
        '''
        prices = self.prices_1d

        start = max(0, idx - window)
        hist = prices[start:idx + 1]  #plus current step maybe change?
        current = prices[idx]

        return float(np.mean(hist <= current)) #What fraction of the recent prices are less than or equal to the current price?

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

    def reward_shape(self, profit, price, volume, volume_to_MWh, action):
        # aciton = 1 pump
        #action -1 sell
        price_current_volume = (volume * volume_to_MWh * price)

        if action > 0:
            return profit + price_current_volume
        else:
            return profit - price_current_volume

    def calculate_energy_in_tank_in_eu(self, volume, price):
        return (volume * self.env.volume_to_MWh * price)

    def calculate_energy_in_tank(self, volume):
        return (volume * self.env.volume_to_MWh)


    def create_Q_table(self):
        self.state_space = len(self.bin_size) - 1
        # Initialize all values in the Q-table to zero

        '''
        ToDo:
        Initialize a zero matrix of dimension state_space * state_space * action_space and call it self.Qtable!
        '''

        # Solution:
        self.Qtable = np.zeros((
            self.bin_size[0],  # Volume bins
            self.bin_size[1],  # Price bins
            self.bin_size[2],  # Hour bins
            self.action_space  # Actions
        ))
        #self.Qtable = np.zeros(tuple(self.bin_size) + (self.action_space,))

    def evaluate_greedy(self, steps_to_plot: int = 48, window: int = 48):
        """
        Run one greedy episode (epsilon=0) and plot the first `steps_to_plot` timesteps.

        Plots:
          1) price + action markers
          2) percentile + action markers
          3) energy in tank
        """
        obs, _ = self.env.reset()
        done = False

        # reset inventory trackers to match your training logic
        _, real_price, *_ = self.env.observation()
        self.energy_in_tank = self.calculate_energy_in_tank(obs[0])
        self.money_in_tank = self.calculate_energy_in_tank_in_eu(obs[0], real_price)

        # logging
        log_price = []
        log_pct = []
        log_action = []
        log_energy = []
        log_money = []

        total_env_reward = 0.0
        t = 0

        # set percentile feature for first state
        idx = self.env.counter
        obs[1] = self.rolling_percentile(idx, window=window)
        state = self.discretize_state(obs)

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
            if t < steps_to_plot:
                log_price.append(float(price_before))
                log_action.append(float(action))

                log_energy.append(float(self.energy_in_tank))
                log_money.append(float(self.money_in_tank))
                log_pct.append(float(obs[1]))  # percentile of current state (before step)

            # prepare next state
            idx = self.env.counter
            next_obs[1] = self.rolling_percentile(idx, window=window)
            obs = next_obs
            state = self.discretize_state(obs)

            t += 1

        # --- plots (first steps_to_plot only) ---
        x = np.arange(len(log_price))

        # helper masks
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

        # Plot 2: percentile + action markers
        plt.figure(figsize=(10, 4))
        plt.plot(x, log_pct)
        if pumps: plt.scatter(pumps, [log_pct[i] for i in pumps], marker="^")
        if sells: plt.scatter(sells, [log_pct[i] for i in sells], marker="v")
        if holds: plt.scatter(holds, [log_pct[i] for i in holds], marker="o", s=20)
        plt.title("Greedy policy (first steps) — Percentile + actions")
        plt.xlabel("timestep")
        plt.ylabel("percentile (0..1)")
        plt.ylim(-0.05, 1.05)
        plt.show()

        # Plot 3: energy in tank
        plt.figure(figsize=(10, 4))
        plt.plot(x, log_energy)
        plt.title("Greedy policy (first steps) — Energy in tank")
        plt.xlabel("timestep")
        plt.ylabel("energy (MWh-equivalent)")
        plt.show()

        print("Greedy evaluation (env reward):", total_env_reward)
        return total_env_reward

    def train(self, simulations, learning_rate, epsilon=0.05, epsilon_decay=1000, adaptive_epsilon=True,
              adapting_learning_rate=False):

        '''
        Params:

        simulations = number of episodes of a game to run
        learning_rate = learning rate for the update equation
        epsilon = epsilon value for epsilon-greedy algorithm
        epsilon_decay = number of full episodes (games) over which the epsilon value will decay to its final value
        adaptive_epsilon = boolean that indicates if the epsilon rate will decay over time or not
        adapting_learning_rate = boolean that indicates if the learning rate should be adaptive or not

        '''

        # Initialize variables that keep track of the rewards

        self.rewards = []
        self.rewards_official = []
        self.average_rewards = []
        self.average_rewards_official = []
        self.money_in_tank = 0.0 # how much money is in total spend on the energy in the tank
        self.energy_in_tank = 0.0 # how much energy is in the tank


        # Call the Q table function to create an initialized Q table
        self.create_Q_table()

        # Set epsilon rate, epsilon decay and learning rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        # Set start epsilon, so here we want a starting exploration rate of 1
        self.epsilon_start = 1
        self.epsilon_end = 0.01

        # If we choose adaptive learning rate, we start with a value of 1 and decay it over time!
        if adapting_learning_rate:
            self.learning_rate = 1

        for i in range(simulations):

            print(f'Please wait, the algorithm is learning! The current simulation is {i}')
            # Initialize the state
            state, _ = self.env.reset()

            _, real_price, *_ = self.env.observation()

            idx = self.env.counter  #should be 0 right after reset
            print("this numer should be 0", idx)
            state[1] = self.rolling_percentile(idx, window=48)

            self.cash = 0.0

            self.money_in_tank = self.calculate_energy_in_tank_in_eu(state[0], real_price)
            self.energy_in_tank = self.calculate_energy_in_tank(state[0])

            #print(state)
            # Set a variable that flags if an episode has terminated
            done = False

            # Discretize the state space

            state = self.discretize_state(state)

            # Set the rewards to 0
            total_rewards = 0           #shaped
            total_reward_official = 0    #not shaped

            # If adaptive epsilon rate
            if adaptive_epsilon:
                self.epsilon = np.interp(i, [0, self.epsilon_decay], [self.epsilon_start, self.epsilon_end])

                # Logging just to check it decays as we want it to do, we just print out the first three statements
                if i % 500 == 0 and i <= 1500:
                    print(f"The current epsilon rate is {self.epsilon}")

            # Loop until an episode has terminated
            while not done:
                # Convert state list to tuple for indexing
                state_tuple = tuple(state)

                # Epsilon-greedy action selection
                if np.random.uniform(0, 1) < self.epsilon:
                    action_index = np.random.randint(self.action_space)
                else:
                    action_index = int(np.argmax(self.Qtable[state_tuple]))

                action = float(self.actions[action_index])

            # Step environment
                volume_before, price_before, *_ = self.env.observation()
                inv_before = 0.9 * price_before * self.energy_in_tank - self.money_in_tank


                # Step environment
                next_state_raw, reward_official, terminated, truncated, info = self.env.step(action)

                volume_after, price_after, *_ = self.env.observation()


                # Energy change
                delta_volume = volume_after - volume_before
                delta_energy = delta_volume * self.env.volume_to_MWh

                idx = self.env.counter
                next_state_raw[1] = self.rolling_percentile(idx, window=48)

                if delta_energy > 0:  # pump (buy)

                    self.energy_in_tank += delta_energy
                    self.money_in_tank += (1/0.8) * price_before * delta_energy
                    reward = 0.0

                elif delta_energy < 0: # sell
                    energy_sold = -delta_energy


                    if self.energy_in_tank > 1e-9: #to not devide by 0
                        avg_price_tank = self.money_in_tank / self.energy_in_tank
                    else:
                        avg_price_tank = 0.0

                    cost_removed = energy_sold * avg_price_tank

                    revenue = 0.9 * price_before * energy_sold

                    reward = revenue - cost_removed

                    self.energy_in_tank -= energy_sold
                    self.money_in_tank -= cost_removed

                    #avoid little bit left over
                    if self.energy_in_tank <= 1e-9:
                        self.energy_in_tank = 0.0
                        self.money_in_tank = 0.0

                else:
                    reward = 0.0

                #add a bit of value for invetory

                inv_after = 0.9 * price_after * self.energy_in_tank - self.money_in_tank

                reward += 0.2 * (inv_after - inv_before)       #TUNE


                done = terminated or truncated

                # sell everything at end of episode otherwise it will learn the wrong thing
                if done and self.energy_in_tank > 1e-9:
                    avg_price_tank = self.money_in_tank / self.energy_in_tank
                    liquidation_profit = 0.9 * self.energy_in_tank * (price_after - avg_price_tank)
                    reward += liquidation_profit
                    self.energy_in_tank = 0.0
                    self.money_in_tank = 0.0

                # Discretize next state
                next_state = self.discretize_state(next_state_raw)
                next_state_tuple = tuple(next_state)

                reward = float(reward)
                # TD Target: reward + gamma * max(Q[next_state])
                Q_target = reward + self.discount_rate * np.max(self.Qtable[next_state_tuple])

                # TD Update
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
            # Calculate the average score over 100 episodes
            if i % 10 == 0:
                self.average_rewards.append(np.mean(self.rewards))
                self.average_rewards_official.append(np.mean(self.rewards_official))

                # Initialize a new reward list, as otherwise the average values would reflect all rewards!
                self.rewards = []
                self.rewards_official = []

        print('The simulation is done!')

        np.save("qtable.npy", self.Qtable)
        print("Q-table saved to qtable.npy")


agent_standard_greedy = QAgent()
agent_standard_greedy.train(
    simulations=1000,    # Increase this! 50 is too low for 2,400 states
    learning_rate=0.03,   # Lower LR is more stable for Q-tables
    epsilon=1.0,          # Start at 100% exploration
    epsilon_decay=800,   # Decay slowly over 80% of training
    adaptive_epsilon=True
)
agent_standard_greedy.evaluate_greedy()
agent_standard_greedy.visualize_rewards()

