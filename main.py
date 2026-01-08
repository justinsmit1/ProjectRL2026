from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class IntradayWaveHeuristic:
    """
    Action in [-1, 1]:
      +1 = pump/buy max
      -1 = sell max
       0 = hold

    Learns an intraday baseline: mean[hour], std[hour] with EMA updates.
    Uses z-score relative to that hour's expected price.
    Tracks a simple position in [-1, 1] to avoid flipping constantly.
    """

    def __init__(
        self,
        alpha=0.03,           # EMA speed for hourly mean/std
        buy_z=1.0,            # buy when price is <= mean - buy_z*std
        sell_z=1.0,           # sell when price is >= mean + sell_z*std
        min_std_frac=0.002,   # floor on std as fraction of mean to avoid divide-by-zero
        step=0.25,            # how aggressively to change action/position
        cooldown=1            # min hours between flips
    ):
        self.alpha = alpha
        self.buy_z = buy_z
        self.sell_z = sell_z
        self.min_std_frac = min_std_frac
        self.step = step
        self.cooldown = cooldown

        self.mean = np.full(24, np.nan, dtype=float)
        self.var = np.full(24, np.nan, dtype=float)

        self.position = 0.0
        self.last_action = 0.0
        self.last_trade_t = -10**9
        self.t = 0

    def _update_hour_stats(self, hour, price):
        h = int(hour) % 24
        a = self.alpha

        if np.isnan(self.mean[h]):
            self.mean[h] = price
            self.var[h] = (0.01 * price) ** 2  # small initial variance
            return

        # EMA mean
        prev_mean = self.mean[h]
        new_mean = (1 - a) * prev_mean + a * price
        self.mean[h] = new_mean

        # EMA variance (of deviations)
        dev = price - new_mean
        prev_var = self.var[h]
        new_var = (1 - a) * prev_var + a * (dev * dev)
        self.var[h] = new_var

    def act(self, observation):
        # obs = [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
        volume, price, hour, dow, doy, month, year = observation
        h = int(hour) % 24

        # Update memory (hourly baseline)
        self._update_hour_stats(h, price)

        mu = self.mean[h]
        std = np.sqrt(max(self.var[h], 0.0))

        # safety floor on std
        std_floor = max(self.min_std_frac * mu, 1e-9)
        std = max(std, std_floor)

        z = (price - mu) / std

        # Optional: volume gating (trade less when volume is super low)
        # Keep it simple: if volume is tiny, reduce action size.
        vol_scale = 1.0
        if volume is not None:
            vol_scale = 0.5 if volume <= 0 else 1.0  # customize if you have a better volume scale

        # Cooldown to reduce flip-flopping
        can_trade = (self.t - self.last_trade_t) >= self.cooldown

        action = 0.0

        # If very cheap for this hour -> pump
        if z <= -self.buy_z and can_trade:
            # move position toward +1
            self.position = min(1.0, self.position + self.step)
            action = +self.step

        # If very expensive for this hour -> sell
        elif z >= self.sell_z and can_trade:
            # move position toward -1
            self.position = max(-1.0, self.position - self.step)
            action = -self.step

        else:
            action = 0.0

        # If we traded, mark time
        if action != 0.0:
            self.last_trade_t = self.t
            self.last_action = action

        self.t += 1

        # Scale action by volume gate
        return float(np.clip(action * vol_scale, -1.0, 1.0))



parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='DATA/validate.xlsx')  # Path to the excel file with the test data
args = parser.parse_args()

env = HydroElectric_Test(path_to_test_data=args.excel_file)
df = pd.read_excel(args.excel_file)
total_reward = []
cumulative_reward = []
heur = IntradayWaveHeuristic(alpha=0.03, buy_z=1.0, sell_z=1.0, step=0.25, cooldown=1)

observation = env.observation()

import numpy as np



def heuristic_action1(observation):
    # obs =[volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    volume, price, hour, dow, doy, month, year = observation

    if 0 <= hour <= 6:
        return 1.0  # pump
    elif 17 <= hour <= 21:
        return -1.0  # sell
    else:
        return 0.0  # hold


def heuristic_action2(observation):
    # obs =[volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    volume, price, hour, dow, doy, month, year = observation

    if 9 <= hour <= 20:
        return -1.0  # pump
    else:
        return 1


def heuristic_action3(observation):
    # obs =[volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    volume, price, hour, dow, doy, month, year = observation

    if price >= 100:
        return -0.5

    if 3 <= hour <= 7:
        return 0.5  # pump
    elif 11 <= hour <= 14:
        return -0.33  # sell
    else:
        return 0.0  # hold




def heuristic_action4(observation):
    # obs =[volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    volume, price, hour, dow, doy, month, year = observation

    x = np.arange(0, 24)

    # Time-based sinusoidal model
    baseline = 0.0        # center around zero for easier combination
    amplitude = 1.0
    period = 24
    phase_shift = -6

    y = baseline + amplitude * np.sin(2 * np.pi * (x + phase_shift) / period)

    time_signal = y[int(hour) - 1]  # [-1, 1]

    # ---- PRICE INFLUENCE ----
    # Expected price range (tune these)
    min_price = 10
    max_price = 100

    # Normalize price to [0, 1]
    price_norm = np.clip((price - min_price) / (max_price - min_price), 0, 1)


    # Final heuristic
    action = time_signal * (price / 150)
    #
    # print(
    #     f"hour={hour}, time_signal={time_signal:.3f}, "
    #     f"price={price}, price_mult={price_multiplier:.3f}, action={action:.3f}"
    # )

    return action





    if (10 <= hour <= 12) or (19 <= hour <= 21):
        return -0.5  # pump
    elif 0 <= hour <= 7:
        return 0.7  # sell
    else:
        return 0.0  # hold


for i in range(730 * 24 - 1):  # Loop through 2 years -> 730 days * 24 hours
    # Choose a random action between -1 (full capacity sell) and 1 (full capacity pump)

    #   action = env.continuous_action_space.sample()
    action = heur.act(observation)

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






# import numpy as np
# import matplotlib.pyplot as plt
#
# # X values (e.g., time or index)
# x = np.arange(0, 24)
#
# # Sinusoidal model (adjust parameters as needed)
# baseline = 50          # vertical offset (mean value)
# amplitude = 10         # height of oscillation
# period = 24            # full cycle length
# phase_shift = -6       # horizontal shift
#
# y = baseline + amplitude * np.sin(2 * np.pi * (x + phase_shift) / period)
#
# # Optional error bars (similar variability to your plot)
# error = 10 + 20 * np.abs(np.sin(2 * np.pi * x / period))
#
# # Plot
# plt.figure(figsize=(10, 4))
# plt.errorbar(x, y, yerr=error, fmt='o-', capsize=4)
# plt.xlabel("X")
# plt.ylabel("Value")
# plt.title("Sinusoidal Approximation of the Given Graph")
# plt.grid(True)
# plt.tight_layout()
# plt.show()