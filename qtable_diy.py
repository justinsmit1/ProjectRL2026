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

    def __init__(self, env_name, discount_rate=0.99, bin_size=20):

        '''
        Params:

        env_name = name of the specific environment that the agent wants to solve
        discount_rate = discount rate used for future rewards
        bin_size = number of bins used for discretizing the state space

        '''

        # create an environment
        self.env = HydroElectric_Test(path_to_test_data = "DATA/validate.xlsx")

        # Set the discount rate
        self.discount_rate = discount_rate

        self.action_space = self.env.discrete_action_space.n

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

        self.bin_size = [10, 8, 6, 5]
        self.day_type_bins = np.array([-0.1, 4.5, 6.1])

        # Manually define meaningful bins
        self.bins = [
            np.linspace(0, self.env.max_volume, self.bin_size[0]),  # Volume
            np.linspace(self.low[1], self.high[1], self.bin_size[1]),  # Price
            np.linspace(1, 24, self.bin_size[2]),  # Hour (every 4 hours)
            self.day_type_bins  # Day of week
        ]

    def discretize_state(self, state):

        '''
        Params:
        state = state observation that needs to be discretized


        Returns:
        discretized state
        '''
        # Now we can make use of the function np.digitize and bin it
        day_of_week = state[3]
        if day_of_week < 5:
            state[3] = 0  # Weekday
        else:
            state[3] = 1  # Weekend

        digitized_state = []
        for i in range(len(self.bins)):
            digitized_state.append(np.digitize(state[i], self.bins[i]) - 1)

        # Returns the discretized state from an observation
        return digitized_state

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
            len(self.day_type_bins) - 1,  # Day-type bins
            self.action_space  # Actions
        ))
        #self.Qtable = np.zeros(tuple(self.bin_size) + (self.action_space,))

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
        self.average_rewards = []

        # Call the Q table function to create an initialized Q table
        self.create_Q_table()

        # Set epsilon rate, epsilon decay and learning rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        # Set start epsilon, so here we want a starting exploration rate of 1
        self.epsilon_start = 1
        self.epsilon_end = 0.05

        # If we choose adaptive learning rate, we start with a value of 1 and decay it over time!
        if adapting_learning_rate:
            self.learning_rate = 1

        for i in range(simulations):

            print(f'Please wait, the algorithm is learning! The current simulation is {i}')
            # Initialize the state
            state = self.env.reset()[0]  # reset returns a dict, need to take the 0th entry.
            #print(state)
            # Set a variable that flags if an episode has terminated
            done = False

            # Discretize the state space

            state = self.discretize_state(state)

            # Set the rewards to 0
            total_rewards = 0

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
                    action_index = self.env.discrete_action_space.sample()
                else:
                    action_index = np.argmax(self.Qtable[state_tuple])

                action = action_index - 1

                # Step environment
                next_state_raw, reward, terminated, truncated, info = self.env.step(action)

                # print(f"Step Reward: {reward} | Info: {info}")
                done = terminated or truncated

                # Discretize next state
                next_state = self.discretize_state(next_state_raw)
                next_state_tuple = tuple(next_state)

                # TD Target: reward + gamma * max(Q[next_state])
                Q_target = reward + self.discount_rate * np.max(self.Qtable[next_state_tuple])

                # TD Update
                current_q = self.Qtable[state_tuple][action]
                self.Qtable[state_tuple][action] = current_q + self.learning_rate * (Q_target - current_q)

                total_rewards += reward
                state = next_state

            if adapting_learning_rate:
                self.learning_rate = self.learning_rate / np.sqrt(i + 1)

            self.rewards.append(total_rewards)
            print(f'Total reward: {np.mean(self.rewards)}')
            # Calculate the average score over 100 episodes
            if i % 100 == 0:
                self.average_rewards.append(np.mean(self.rewards))

                # Initialize a new reward list, as otherwise the average values would reflect all rewards!
                self.rewards = []

        print('The simulation is done!')

agent_standard_greedy = QAgent("test")
agent_standard_greedy.train(
    simulations=15000,    # Increase this! 50 is too low for 2,400 states
    learning_rate=0.01,   # Lower LR is more stable for Q-tables
    epsilon=1.0,          # Start at 100% exploration
    epsilon_decay=8000,   # Decay slowly over 80% of training
    adaptive_epsilon=True
)
l = 1
