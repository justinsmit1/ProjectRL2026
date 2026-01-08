from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='DATA/validate.xlsx')
parser.add_argument('--num_runs', type=int, default=100)
args = parser.parse_args()

NUM_STEPS = 730 * 24  # 2 years in hours
total_rewards = []

for run in range(args.num_runs):
    env = HydroElectric_Test(path_to_test_data=args.excel_file)

    observation = env.observation()
    episode_reward = 0.0

    for step in range(NUM_STEPS - 1):
        # Random action (replace with RL agent if needed)
        action = env.continuous_action_space.sample()

        next_observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        observation = next_observation

        if terminated or truncated:
            break

    total_rewards.append(episode_reward)
    print(f"Run {run + 1}/{args.num_runs} - Total reward: {episode_reward:.2f}")

# Convert to numpy array for analysis
total_rewards = np.array(total_rewards)

# ---- Basic statistics ----
print("\n===== Experiment Results =====")
print(f"Mean total reward: {total_rewards.mean():.2f}")
print(f"Std total reward : {total_rewards.std():.2f}")
print(f"Min total reward : {total_rewards.min():.2f}")
print(f"Max total reward : {total_rewards.max():.2f}")

# ---- Plot distribution of total rewards ----
plt.figure()
plt.hist(total_rewards, bins=20)
plt.xlabel("Total Reward per Run")
plt.ylabel("Frequency")
plt.title("Distribution of Total Rewards over 100 Runs")
plt.show()
