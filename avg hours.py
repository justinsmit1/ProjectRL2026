from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='validate.xlsx') # Path to the excel file with the test data
args = parser.parse_args()

env = HydroElectric_Test(path_to_test_data=args.excel_file)
total_reward = []
cumulative_reward = []

observation = env.observation()

def heuristic_mees3(observation, avg):
    #Takes into account that there is a peak in the winter month later in the day
    #Also uses avrage price to be sure the price is good
    #Include treshholds

    # obs =[volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    treshhold = 0
    volume, price, hour, dow, doy, month, year = observation
    if month in [11, 10, 12, 1, 2]:
        if 2 <= hour <= 7 and price < avg-treshhold:
            return 1.0   # pump
        elif 10 <= hour <= 12 and price > avg+treshhold:
            return -1.0  # sell
        elif 18 <= hour <= 21 and price > avg+treshhold:
            return -1.0  # sell
        else:
            return 0.0   # hold
    else:
        if 2 <= hour <= 7 and price < avg-treshhold:
            return 1.0   # pump
        elif 9 <= hour <= 14 and price > avg+treshhold:
            return -1.0  # sell
        else:
            return 0.0   # hold

prices = []

# --- NEW: profit-day tracking ---
profit_days = 0
day_reward = 0.0
days_total = 0
# -------------------------------

for i in range(730*24 - 1):  # Loop through 2 years -> 730 days * 24 hours
    volume, price, hour, dow, doy, month, year = observation

    prices.append(price)
    last_prices = prices[-24:]
    avg_24h_price = sum(last_prices) / len(last_prices)

    action = heuristic_mees3(observation, avg_24h_price)

    next_observation, reward, terminated, truncated, info = env.step(action)

    total_reward.append(reward)
    cumulative_reward.append(sum(total_reward))

    # --- NEW: accumulate rewards per day ---
    day_reward += reward

    # If we just finished a day (every 24 hours), evaluate profitability
    if (i + 1) % 24 == 0:
        days_total += 1
        if day_reward > 0:
            profit_days += 1
        day_reward = 0.0
    # --------------------------------------

    done = terminated or truncated
    observation = next_observation

    if done:
        break

print('Total reward: ', sum(total_reward))
print(f'Days with profit: {profit_days} / {days_total}')

# Plot the cumulative reward over time
plt.plot(cumulative_reward)
plt.xlabel('Time (Hours)')
plt.ylabel('Cumulative Reward')
plt.show()
