from TestEnv import HydroElectric_Test
import numpy as np


class QAgent:

    def __init__(self, env_factory, discount_rate=0.95, bins=None):
        """
        env_factory: callable that returns a NEW HydroElectric_Test instance
        """

        self.env_factory = env_factory
        self.discount_rate = discount_rate

        # Create one env to read constants
        env = env_factory()

        self.action_space = env.discrete_action_space.n

        # ----------------------------
        # MANUAL STATE BOUNDS
        # ----------------------------
        self.low = np.array([
            0,                              # volume
            np.min(env.price_values),       # price
            1,                              # hour
            0,                              # day_of_week
            1,                              # day_of_year
            1,                              # month
            env.timestamps.dt.year.min()    # year
        ], dtype=np.float32)

        self.high = np.array([
            env.max_volume,
            np.max(env.price_values),
            24,
            6,
            366,
            12,
            env.timestamps.dt.year.max()
        ], dtype=np.float32)

        # ----------------------------
        # DISCRETIZATION
        # ----------------------------
        if bins is None:
            bins = [10, 10, 24, 7, 12, 12, 3]

        self.bins = [
            np.linspace(self.low[i], self.high[i], bins[i])
            for i in range(len(bins))
        ]

        self.state_space = [b - 1 for b in bins]

        self.create_Q_table()

    # ----------------------------
    # DISCRETIZATION
    # ----------------------------
    def discretize_state(self, state):
        return tuple(
            np.clip(
                np.digitize(state[i], self.bins[i]) - 1,
                0,
                self.state_space[i] - 1
            )
            for i in range(len(self.bins))
        )

    # ----------------------------
    # Q TABLE
    # ----------------------------
    def create_Q_table(self):
        self.Qtable = np.zeros((*self.state_space, self.action_space))

    # ----------------------------
    # TRAINING
    # ----------------------------
    def train(
        self,
        simulations,
        learning_rate,
        epsilon=0.1,
        epsilon_decay=1000,
        adaptive_epsilon=False
    ):

        self.learning_rate = learning_rate
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon = epsilon

        for episode in range(simulations):

            # 🔴 NEW ENV PER EPISODE (no reset)
            env = self.env_factory()

            state = env.observation()
            state = self.discretize_state(state)

            done = False
            total_reward = 0

            if adaptive_epsilon:
                self.epsilon = np.interp(
                    episode,
                    [0, epsilon_decay],
                    [self.epsilon_start, self.epsilon_end]
                )

            while not done:

                if np.random.rand() < self.epsilon:
                    action = env.discrete_action_space.sample()
                else:
                    action = np.argmax(self.Qtable[state])

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_state = self.discretize_state(next_state)

                td_target = reward + self.discount_rate * np.max(self.Qtable[next_state])
                td_error = td_target - self.Qtable[state + (action,)]

                self.Qtable[state + (action,)] += self.learning_rate * td_error

                state = next_state
                total_reward += reward

            if episode % 500 == 0:
                print(f"Episode {episode}, reward: {total_reward}")

        print("Training completed.")

    # ----------------------------
    # POLICY EXECUTION
    # ----------------------------
    def play_game(self):
        env = self.env_factory()
        state = self.discretize_state(env.observation())
        done = False

        while not done:
            action = np.argmax(self.Qtable[state])
            state, reward, terminated, truncated, _ = env.step(action)
            state = self.discretize_state(state)
            done = terminated or truncated


agent = QAgent(
    env_factory=lambda: HydroElectric_Test("DATA/validate.xlsx"),
    discount_rate=0.95
)

agent.train(
    simulations=3000,
    learning_rate=0.1,
    adaptive_epsilon=True
)