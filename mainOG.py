from TestEnv import HydroElectric_Test
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='DATA/validate.xlsx') # Path to the excel file with the test data
args = parser.parse_args()

env = HydroElectric_Test(path_to_test_data=args.excel_file)
total_reward = []
cumulative_reward = []

observation = env.observation()

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
    # obs =[volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    volume, price, hour, dow, doy, month, year = observation

    if 9 <= hour <= 20:
        return -1.0   # pump
    else:
        return 1

def heuristic_action3(observation):
    # obs =[volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    volume, price, hour, dow, doy, month, year = observation

    if 3 <= hour <= 7:
        return 1.0   # pump
    elif 11 <= hour <= 14:
        return -1.0  # sell
    else:
        return 0.0   # hold

def heuristic_action4(observation):
    # obs =[volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    volume, price, hour, dow, doy, month, year = observation

    if 3 <= hour <= 7:
        return 1.0   # pump
    elif 11 <= hour <= 14:
        return -1.0  # sell
    else:
        return 0.0   # hold

def heuristic_mees(observation):
    #Takes into account that there is a peak in the winter month later in the day
    # obs =[volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    volume, price, hour, dow, doy, month, year = observation
    if month in [11, 10, 12, 1, 2]:
        if 2 <= hour <= 7:
            return 0.2   # pump
        elif 10 <= hour <= 12:
            return -0.5  # sell
        elif 18 <= hour <= 21:
            return -0.33  # sell
        else:
            return 0.0   # hold
    else:
        if 2 <= hour <= 7:
            return 0.2   # pump
        elif 9 <= hour <= 14:
            return -01.0  # sell
        else:
            return 0.0   # hold

def heuristic_mees2(observation, avg):
    #Takes into account that there is a peak in the winter month later in the day
    #Also uses avrage price to be sure the price is good
    # obs =[volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    volume, price, hour, dow, doy, month, year = observation
    if month in [11, 10, 12, 1, 2]:
        if 2 <= hour <= 7 and price < avg:
            return 1.0   # pump
        elif 10 <= hour <= 12 and price > avg:
            return -1.0  # sell
        elif 18 <= hour <= 21 and price > avg:
            return -1.0  # sell
        else:
            return 0.0   # hold
    else:
        if 2 <= hour <= 7 and price < avg:
            return 1.0   # pump
        elif 9 <= hour <= 14 and price > avg:
            return -1.0  # sell
        else:
            return 0.0   # hold

def heuristic_mees3(observation, avg):
    """
    observation = [volume, price, hour_of_day, day_of_week,
                   day_of_year, month_of_year, year]

    Output range: [-1, 1]
    +1  = max pump
    -1  = max sell
    0   = hold
    """

    threshold = 0
    volume, price, hour, dow, doy, month, year = observation

    # Normalize tank level to [0, 1]
    tank_capacity = volume / 100000
    tank_capacity = max(0.0, min(1.0, tank_capacity))

    # Normalized price difference (scale controls aggressiveness)
    price_scale = avg if avg != 0 else 1.0
    price_diff = (price - avg) / price_scale

    def pump_strength():
        # Stronger when tank is empty AND price is much lower than avg
        return (1.0 - tank_capacity) * max(0.0, -price_diff)

    def sell_strength():
        # Stronger when tank is full AND price is much higher than avg
        return -tank_capacity * max(0.0, price_diff)

    winter_months = [11, 10, 12, 1, 2]

    if month in winter_months:
        if 2 <= hour <= 7 and price < avg - threshold and tank_capacity < 1.0:
            return pump_strength()

        elif ((10 <= hour <= 12) or (18 <= hour <= 21)) \
                and price > avg + threshold and tank_capacity > 0.0:
            return sell_strength()

        else:
            return 0.0

    else:
        if 2 <= hour <= 7 and price < avg - threshold and tank_capacity < 1.0:
            return pump_strength()

        elif 9 <= hour <= 14 and price > avg + threshold and tank_capacity > 0.0:
            return sell_strength()

        else:
            return 0.0

prices = []

for i in range(730*24 -1): # Loop through 2 years -> 730 days * 24 hours
    # Choose a random action between -1 (full capacity sell) and 1 (full capacity pump)
    volume, price, hour, dow, doy, month, year = observation


    prices.append(price)

    last_prices = prices[-24:]
    avg_24h_price = sum(last_prices) / len(last_prices)


 #   action = env.continuous_action_space.sample()
    action = heuristic_mees3(observation, avg_24h_price)

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




