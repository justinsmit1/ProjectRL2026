from TestEnv import HydroElectric_Test
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='validate.xlsx')
args = parser.parse_args()

env = HydroElectric_Test(path_to_test_data=args.excel_file)

def heuristic_action3(observation):
    volume, price, hour, dow, doy, month, year = observation
    if 3 <= hour <= 7:
        return np.array([1.0], dtype=np.float32)    # pump
    elif 11 <= hour <= 14:
        return np.array([-1.0], dtype=np.float32)   # generate/sell
    else:
        return np.array([0.0], dtype=np.float32)    # hold

def run_and_plot_average_day():
    obs, _ = env.reset()

    rows = []
    done = False
    total_reward = 0.0

    while not done:
        action = heuristic_action3(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)

        # Log: hour-of-day is obs[2] (current hour), volume after step is next_obs[0]
        rows.append({
            "hour": int(obs[2]),                      # 1..24
            "price": float(obs[1]),
            "action": float(np.squeeze(action)),      # -1, 0, 1
            "volume_after": float(next_obs[0]),
            "reward": float(reward),
        })

        obs = next_obs

    print("Total reward:", total_reward)

    df = pd.DataFrame(rows)

    # Build "average day" (mean per hour-of-day)
    avg = df.groupby("hour").agg(
        avg_price=("price", "mean"),
        avg_volume=("volume_after", "mean"),
        pump_rate=("action", lambda x: np.mean(x > 0)),
        gen_rate=("action", lambda x: np.mean(x < 0)),
        hold_rate=("action", lambda x: np.mean(x == 0)),
        avg_reward=("reward", "mean"),
    ).reset_index().sort_values("hour")

    # ---------- Plot 1: Avg volume + avg price ----------
    fig, ax1 = plt.subplots()

    ax1.plot(avg["hour"], avg["avg_volume"])
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Average reservoir volume (m³)")
    ax1.set_xticks(range(1, 25))

    ax2 = ax1.twinx()
    ax2.plot(avg["hour"], avg["avg_price"])
    ax2.set_ylabel("Average price")

    plt.title("Average day: reservoir volume and price")
    plt.show()

    # ---------- Plot 2: Action frequency by hour ----------
    plt.figure()
    plt.plot(avg["hour"], avg["pump_rate"], label="Pump frequency")
    plt.plot(avg["hour"], avg["gen_rate"], label="Generate frequency")
    plt.plot(avg["hour"], avg["hold_rate"], label="Hold frequency")
    plt.xlabel("Hour of day")
    plt.ylabel("Fraction of days")
    plt.title("Average day: how often the heuristic pumps/generates")
    plt.xticks(range(1, 25))
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.show()

    # ---------- Plot 3 (optional): Avg reward per hour ----------
    plt.figure()
    plt.plot(avg["hour"], avg["avg_reward"])
    plt.xlabel("Hour of day")
    plt.ylabel("Average reward per hour")
    plt.title("Average day: reward contribution by hour")
    plt.xticks(range(1, 25))
    plt.show()

if __name__ == "__main__":
    run_and_plot_average_day()
