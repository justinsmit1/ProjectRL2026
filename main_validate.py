from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt
import numpy as np
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='validate.xlsx')
parser.add_argument('--qtable', type=str, default='qtable.npy')
parser.add_argument("--bins", type = str, default = 'bins.yaml')
args = parser.parse_args()

env = HydroElectric_Test(path_to_test_data="DATA/validate.xlsx")
#env = HydroElectric_Test(path_to_test_data=args.excel_file)
total_reward = []
cumulative_reward = []
#q_table = np.load(args.qtable)
q_table = np.load("qtable_mees.npy")

# actions = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
# action_space = len(actions)

actions = np.array([-1.0, 0, 1.0], dtype=np.float32)
action_space = len(actions)


def load_bins(path="bins.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    bins = [
        np.linspace(spec["min"], spec["max"], spec["size"])
        for spec in config["bins"].values()
    ]

    bin_size = [spec["size"] for spec in config["bins"].values()]

    return bins, bin_size

bins, bin_size = load_bins(path = args.bins)
#
# bin_size = [3, 5, 8, 2, 4]
# bins = [
#             np.linspace(0, 100000, bin_size[0]),  # Volume
#             #np.linspace(0, self.high[1], self.bin_size[1]),  # Price
#             np.linspace(0, 1, bin_size[1]),  # Price
#             np.linspace(1, 24, bin_size[2]),  # Hour (every 4 hours)
#             np.linspace(0, 1, bin_size[3]),  # weekend
#             np.linspace(0, 3, bin_size[4])  # seasons
#         ]

# def discretize_state(bins, state):
#     digitized_state = []
#     for i in range(len(bins)):
#         raw_idx = np.digitize(state[i], bins[i]) - 1
#         safe_idx = int(np.clip(raw_idx, 0, len(bins[i]) - 2))
#         digitized_state.append(safe_idx)
#     return digitized_state


def discretize_state(bins, state):
    '''
    Params:
    state = state observation that needs to be discretized

    Returns:
    discretized state
    '''

    # [dam_level, price, int(hour), int(day_of_week), int(day_of_year), int(month), int(year)]
    month = state[5]

    match month:
        case 4 | 5 | 6 | 7 | 8 | 9:
            state[5] = 0  # "summer"
        case 11 | 12 | 1 | 2 | 3:
            state[5] = 1  # "winter"

    # match month:
    #     case 12 | 1 | 2:
    #         state[5] = 0  # "winter"
    #     case 3 | 4 | 5:
    #         state[5] = 1  # "spring"
    #     case 6 | 7 | 8:
    #         state[5] = 2  # "summer"
    #     case 9 | 10 | 11:
    #         state[5] = 3  # "autumn"

    # Now we can make use of the function np.digitize and bin it

    # day_of_week = state[3]
    # if day_of_week < 5:
    #     state[3] = 0  # Weekday
    # else:
    #     state[3] = 1  # Weekend

    digitized_state = []
    for i in range(len(bins)):
        digitized_state.append(np.digitize(state[i], bins[i]) - 1)

    # Returns the discretized state from an observation
    return digitized_state

observation = env.observation()
for i in range(730*24 -1): # Loop through 2 years -> 730 days * 24 hours
    # Choose a random action between -1 (full capacity sell) and 1 (full capacity pump)

    # discretize states
    state = discretize_state(bins, observation)
    state_tuple = tuple(state)
    print(state)
    action_index = int(np.argmax(q_table[state_tuple]))

    action = float(actions[action_index])
    # Or choose an action based on the observation using your RL agent!:
    # action = RL_agent.act(observation)
    # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    next_observation, reward, terminated, truncated, info = env.step(action)
    total_reward.append(reward)
    cumulative_reward.append(sum(total_reward))

    done = terminated or truncated
    observation = next_observation

    if done:
        print('Total reward: ', sum(total_reward))
        # Plot the cumulative reward over time
        plt.plot(cumulative_reward)
        plt.xlabel('Time (Hours)')
        plt.show()




