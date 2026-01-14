
from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '--excel_file',
    type=str,
    default='DATA/validate.xlsx',
    help='Path to the excel file with the test data'
)
args = parser.parse_args()

# Initialize environment
env = HydroElectric_Test(path_to_test_data=args.excel_file)

total_reward = []
cumulative_reward = []

# Initial observation
observation = env.observation()


class QLearningDam:
    def __init__(self, discount_rate = 0.95, bin_size = 20):
        self.env = HydroElectric_Test(path_to_test_data='DATA/train.xlsx')
        self.discount_rate = discount_rate

        self.action_space = self.env.discrete_action_space.n

        self.bin_size = bin_size

        self.volume_low = 0
        self.volume_high = 100000

        self.bin_volume = np.linspace(self.volume_low, self.volume_high, self.bin_size)

        self.bins = [self.bin_volume]

    def discretize_state(self, state):

        '''
        Params:
        state = state observation that needs to be discretized


        Returns:
        discretized state
        '''
        # Now we can make use of the function np.digitize and bin it
        # self.state = state
        #
        # # Create an empty state
        # digitized_state = []
        #
        # for i in range(len(self.bins)):
        #     digitized_state.append(np.digitize(self.state[i], self.bins[i]) - 1)
        #
        # # Returns the discretized state from an observation
        # return digitized_state
        digitized_state = []
        for i in range(len(self.bins)):
            digitized_state.append(np.digitize(self.state[i], self.bins[i]) - 1)
        return digitized_state

    # def discretize_state(self, state):
    #     # state is just a scalar: volume
    #     digitized_state = np.digitize(state[0], self.bin_volume) - 1  # subtract 1 to get 0-index
    #     return digitized_state

    def create_Q_table(self):
        self.state_space = self.bin_size - 1
        # Initialize all values in the Q-table to zero

        '''
        ToDo:
        Initialize a zero matrix of dimension state_space * state_space * action_space and call it self.Qtable!
        '''

        # Solution:
        self.Qtable = np.zeros((self.bin_size, self.action_space))

    def train(self, simulations, learning_rate, epsilon=0.05, epsilon_decay=1000, adaptive_epsilon=False,
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

            if i % 5000 == 0:
                print(f'Please wait, the algorithm is learning! The current simulation is {i}')
            # Initialize the state
            state, _ = self.env.reset()  # reset returns a dict, need to take the 0th entry.

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

                # Pick an action based on epsilon greedy

                '''
                ToDo: Write the if statement that picks a random action
                Tip: Make use of np.random.uniform() and the self.epsilon to make a decision!
                Tip: You can also make use of the method sample() of the self.env.action_space 
                    to generate a random action!
                '''

                # Solution:

                # Pick random action
                if np.random.uniform(0, 1) > 1 - self.epsilon:
                    # This picks a random action from 0,1,2
                    action = self.env.discrete_action_space.sample()


                # Pick a greedy action
                else:
                    action = np.argmax(self.Qtable[state[0], :])

                # Now sample the next_state, reward, done and info from the environment

                next_state, reward, terminated, truncated, info = self.env.step(action)  # step returns 5 outputs
                done = terminated or truncated

                # Now discretize the next_state
                next_state = self.discretize_state(next_state)

                # Target value
                Q_target = (reward + self.discount_rate * np.max(self.Qtable[next_state[0]]))

                # Calculate the Temporal difference error (delta)
                delta = self.learning_rate * (Q_target - self.Qtable[state[0], action])

                # Update the Q-value
                self.Qtable[state[0], action] = self.Qtable[state[0], action] + delta

                # Update the reward and the hyperparameters
                total_rewards += reward
                state = next_state

            if adapting_learning_rate:
                self.learning_rate = self.learning_rate / np.sqrt(i + 1)

            self.rewards.append(total_rewards)

            # Calculate the average score over 100 episodes
            if i % 100 == 0:
                self.average_rewards.append(np.mean(self.rewards))

                # Initialize a new reward list, as otherwise the average values would reflect all rewards!
                self.rewards = []

        print('The simulation is done!')


    def visualize_rewards(self):
        plt.figure(figsize=(7.5, 7.5))
        plt.plot(100 * (np.arange(len(self.average_rewards)) + 1), self.average_rewards)
        plt.axhline(y=-110, color='r', linestyle='-')
        plt.title('Average reward over the past 100 simulations', fontsize=10)
        plt.legend(['Q-learning performance', 'Benchmark'])
        plt.xlabel('Number of simulations', fontsize=10)
        plt.ylabel('Average reward', fontsize=10)

    def play_game(self):
        # Make eval env which renders when taking a step
        eval_env = HydroElectric_Test(path_to_test_data='DATA/validate.xlsx')
        state = eval_env.reset()[0]
        done = False
        # Run the environment for 1 episode
        while not done:
            state = self.discretize_state(state)
            action = np.argmax(self.Qtable[state[0], state[1], :])
            next_state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            state = next_state
        eval_env.close()



simulations = 15000
learning_rate = 0.10

'''
ToDo:
Initialize the Qagent and call the instance agent!
Afterwards,let it train over the number of simulations with the specific learning rate!
'''

#WRITE YOUR CODE HERE!
#With standard epsilon_greedy
agent_standard_greedy = QLearningDam()
agent_standard_greedy.train(simulations, learning_rate)

#
# # Loop through 2 years -> 730 days * 24 hours
# for i in range(730 * 24 - 1):
#     print(env)
#     # Choose a random action between
#     # -1 (full capacity sell) and 1 (full capacity pump)
#     action = env.continuous_action_space.sample()
#
#     # Or choose an action based on the observation using your RL agent:
#     # action = RL_agent.act(observation)
#
#     # Observation format:
#     # [volume, price, hour_of_day, day_of_week,
#     #  day_of_year, month_of_year, year]
#     next_observation, reward, terminated, truncated, info = env.step(action)
#
#     total_reward.append(reward)
#     cumulative_reward.append(sum(total_reward))
#
#     done = terminated or truncated
#     observation = next_observation
#
#     if done:
#         print('Total reward:', sum(total_reward))
#         break
#
# # Plot the cumulative reward over time
# plt.plot(cumulative_reward)
# plt.xlabel('Time (Hours)')
# plt.ylabel('Cumulative Reward')
# plt.show()
