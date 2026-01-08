import numpy as np

# import your env class from the file where you defined it
# Example: if your environment code is in hydro_env.py
from hydro_env import HydroElectric_Test


def time_window_heuristic(obs):
    """
    obs = [dam_level, price, hour, day_of_week, day_of_year, month, year]
    Pump 08-10, Generate 15-17, else idle.
    """
    hour = int(obs[2])  # 1..24

    # Pump between 8 and 10 inclusive
    if 8 <= hour <= 10:
        return np.array([1.0], dtype=np.float32)

    # Generate between 15 and 17 inclusive
    if 15 <= hour <= 17:
        return np.array([-1.0], dtype=np.float32)

    return np.array([0.0], dtype=np.float32)


def run_sim(path_to_xlsx):
    env = HydroElectric_Test(path_to_xlsx)

    obs, info = env.reset()
    total_reward = 0.0

    done = False
    step_i = 0

    while not done:
        action = time_window_heuristic(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        done = terminated or truncated
        step_i += 1

    print("Finished simulation.")
    print("Steps:", step_i)
    print("Total reward (profit):", total_reward)
    print("Final reservoir volume (m^3):", env.volume)


if __name__ == "__main__":
    # Change this to your file path:
    run_sim("train.xlsx")
