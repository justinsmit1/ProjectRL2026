from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt

# -----------------------------
# Heuristics
# -----------------------------
def heuristic_action1(observation):
    # obs =[volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    volume, price, hour, dow, doy, month, year = observation
    if 0 <= hour <= 6:
        return 1.0   # pump
    elif 17 <= hour <= 21:
        return -1.0  # sell
    else:
        return 0.0   # hold

def heuristic_action2(observation):
    volume, price, hour, dow, doy, month, year = observation
    if 9 <= hour <= 20:
        return -1.0
    else:
        return 1.0

def heuristic_action3(observation):
    volume, price, hour, dow, doy, month, year = observation
    if 3 <= hour <= 7:
        return 1.0   # pump
    elif 11 <= hour <= 14:
        return -1.0  # sell
    else:
        return 0.0   # hold

def heuristic_action4(observation):
    volume, price, hour, dow, doy, month, year = observation
    if 3 <= hour <= 7:
        return 1.0   # pump
    elif 11 <= hour <= 14:
        return -1.0  # sell
    else:
        return 0.0   # hold

def heuristic_mees(observation):
    # Takes into account that there is a peak in the winter month later in the day
    volume, price, hour, dow, doy, month, year = observation
    if month in [11, 10, 12, 1, 2]:
        if 2 <= hour <= 7:
            return 1.0   # pump
        elif 10 <= hour <= 12:
            return -1.0  # sell
        elif 18 <= hour <= 21:
            return -1.0  # sell
        else:
            return 0.0   # hold
    else:
        if 2 <= hour <= 7:
            return 1.0   # pump
        elif 9 <= hour <= 14:
            return -1.0  # sell
        else:
            return 0.0   # hold

def heuristic_mees2(observation, avg):
    # Winter peak later in day + compare against 24h average price
    volume, price, hour, dow, doy, month, year = observation
    if month in [11, 10, 12, 1, 2]:
        if 2 <= hour <= 7 and price < avg:
            return 1.0
        elif 10 <= hour <= 12 and price > avg:
            return -1.0
        elif 18 <= hour <= 21 and price > avg:
            return -1.0
        else:
            return 0.0
    else:
        if 2 <= hour <= 7 and price < avg:
            return 1.0
        elif 9 <= hour <= 14 and price > avg:
            return -1.0
        else:
            return 0.0

def heuristic_mees3(observation, avg, threshold=0.0):
    # Winter peak later in day + 24h avg + threshold
    volume, price, hour, dow, doy, month, year = observation
    if month in [11, 10, 12, 1, 2]:
        if 2 <= hour <= 7 and price < avg - threshold:
            return 1.0
        elif 10 <= hour <= 12 and price > avg + threshold:
            return -1.0
        elif 18 <= hour <= 21 and price > avg + threshold:
            return -1.0
        else:
            return 0.0
    else:
        if 2 <= hour <= 7 and price < avg - threshold:
            return 1.0
        elif 9 <= hour <= 14 and price > avg + threshold:
            return -1.0
        else:
            return 0.0

def heuristic_mees4(observation, avg, threshold=0.0):
    # Winter peak later in day + 24h avg + threshold
    volume, price, hour, dow, doy, month, year = observation
    if month in [11, 10, 12, 1, 2]:
        if 1 <= hour <= 6 and price < avg - threshold:
            return 1.0
        elif 10 <= hour <= 12 and price > avg + threshold:
            return -1.0
        elif 18 <= hour <= 21 and price > avg + threshold:
            return -1.0
        else:
            return 0.0
    else:
        if 1 <= hour <= 6 and price < avg - threshold:
            return 1.0
        elif 9 <= hour <= 14 and price > avg + threshold:
            return -1.0
        else:
            return 0.0


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_day_from_logs(logs, target_year, target_doy, title):
    """
    Plot one day (up to 24 hours) of:
      - price (line)
      - action (step)
      - volume (dashed)
    selected by (year, day_of_year).
    """
    years = logs["year"]
    doys = logs["doy"]
    hours = logs["hour"]
    prices = logs["price"]
    actions = logs["action"]
    volumes = logs["volume"]

    # find indices matching the requested day
    idx = [i for i in range(len(actions)) if years[i] == target_year and doys[i] == target_doy]
    if not idx:
        print(f"[plot_day] No data for year={target_year}, doy={target_doy}")
        return

    # Prefer starting at hour==0 (start of day) if present
    start_candidates = [i for i in idx if hours[i] == 0]
    start = start_candidates[0] if start_candidates else idx[0]
    day_idx = list(range(start, min(start + 24, len(actions))))

    h = [hours[i] for i in day_idx]
    p = [prices[i] for i in day_idx]
    a = [actions[i] for i in day_idx]
    v = [volumes[i] for i in day_idx]

    fig, ax1 = plt.subplots()
    ax1.plot(h, p, label = "Price", color = "orange")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Price")

    ax2 = ax1.twinx()
    ax2.plot(h, v, linestyle="--", label = "Reservoir volume")
    ax2.set_ylabel("Action (step) / Volume (dashed)")

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title(title)
    plt.xticks(range(0, 24, 2))
    plt.show()


def auto_pick_day_by_month(logs, target_month):
    """
    Pick the first day-of-year (doy) we see for a given month.
    Returns (year, doy) or None if not found.
    """
    for y, m, d in zip(logs["year"], logs["month"], logs["doy"]):
        if m == target_month:
            return (y, d)
    return None

def compute_daily_profit(logs, target_year, target_doy):
    """
    Returns total profit (sum of rewards) for the given (year, doy).
    """
    years = logs["year"]
    doys = logs["doy"]
    hours = logs["hour"]
    rewards = logs["reward"]

    idx = [i for i in range(len(hours)) if years[i] == target_year and doys[i] == target_doy]
    if not idx:
        return None

    # Prefer a full day starting at hour 0
    start_candidates = [i for i in idx if hours[i] == 0]
    start = start_candidates[0] if start_candidates else idx[0]

    day_idx = list(range(start, min(start + 24, len(rewards))))

    return sum(rewards[i] for i in day_idx)



# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_file', type=str, default='validate.xlsx', help="Path to excel file with test data")
    parser.add_argument('--threshold', type=float, default=0.0, help="Price threshold for heuristic_mees3")
    parser.add_argument('--winter_doy', type=int, default=40, help="Explicit winter day-of-year to plot (optional)")
    parser.add_argument('--summer_doy', type=int, default=210, help="Explicit summer day-of-year to plot (optional)")
    parser.add_argument('--winter_month', type=int, default=1, help="Winter month to auto-pick (default Jan=1)")
    parser.add_argument('--summer_month', type=int, default=7, help="Summer month to auto-pick (default Jul=7)")
    args = parser.parse_args()

    env = HydroElectric_Test(path_to_test_data=args.excel_file)

    total_reward = []
    cumulative_reward = []

    observation = env.observation()

    # Logs for plotting behavior
    logs = {
        "volume": [],
        "price": [],
        "hour": [],
        "dow": [],
        "doy": [],
        "month": [],
        "year": [],
        "action": [],
        "reward": [],
        "cum_reward": []
    }

    prices_window = []  # for 24h rolling average

    # Run full episode (2 years -> 730 days * 24 hours - 1)
    for t in range(730 * 24 - 1):
        volume, price, hour, dow, doy, month, year = observation

        # update rolling window
        prices_window.append(price)
        last_prices = prices_window[-24:]
        avg_24h_price = sum(last_prices) / len(last_prices)

        # choose action (your algorithm)
        action = heuristic_mees3(observation, avg_24h_price, threshold=args.threshold)

        next_observation, reward, terminated, truncated, info = env.step(action)

        total_reward.append(reward)
        cumulative_reward.append(sum(total_reward))

        # log everything
        logs["volume"].append(volume)
        logs["price"].append(price)
        logs["hour"].append(hour)
        logs["dow"].append(dow)
        logs["doy"].append(doy)
        logs["month"].append(month)
        logs["year"].append(year)
        logs["action"].append(action)
        logs["reward"].append(reward)
        logs["cum_reward"].append(cumulative_reward[-1])

        done = terminated or truncated
        observation = next_observation

        if done:
            break

    print("Total reward:", sum(total_reward))

    # -----------------------------
    # Choose winter & summer days to plot
    # -----------------------------
    # If user provided doy, use those. Otherwise auto-pick a day from winter_month / summer_month.
    if args.winter_doy is not None:
        winter = (logs["year"][0], args.winter_doy)
    else:
        winter = auto_pick_day_by_month(logs, args.winter_month)

    if args.summer_doy is not None:
        summer = (logs["year"][0], args.summer_doy)
    else:
        summer = auto_pick_day_by_month(logs, args.summer_month)

    if winter is None:
        print(f"Could not find any data for winter_month={args.winter_month}")
    else:
        wy, wd = winter
        plot_day_from_logs(logs, wy, wd, title=f"Behavior on winter day (year={int(wy)}, doy={wd})")
        winter_profit = compute_daily_profit(logs, wy, wd)
        print("winter profit", winter_profit)
    if summer is None:
        print(f"Could not find any data for summer_month={args.summer_month}")
    else:
        sy, sd = summer
        plot_day_from_logs(logs, sy, sd, title=f"Behavior on summer day (year={int(sy)}, doy={sd})")
        summer_profit = compute_daily_profit(logs, sy, sd)
        print("summer profit: ", summer_profit)

    # Optional: plot cumulative reward for the whole run
    plt.figure()
    plt.plot(logs["cum_reward"])
    plt.xlabel("Time (Hours)")
    plt.ylabel("Cumulative reward")
    plt.title("Cumulative reward over time")
    plt.show()


if __name__ == "__main__":
    main()
