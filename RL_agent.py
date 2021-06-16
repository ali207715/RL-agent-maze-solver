import random
import numpy as np


def output_dict(env, q_table):
    """
        Utilizes the final q_table and outputs a dict containing the optimal policy.

        :param q_table: {nest list}; contains the values of each action for all states.
        :param env: {class instance); contains information about the environment.
        :return: optimal_policy {dict}; key=state:value=optimal action.
    """

    optimal_policy = {}
    x_dims = env.observation_space.spaces[0].n
    y_dims = env.observation_space.spaces[1].n

    for x in range(x_dims):
        for y in range(y_dims):
            action = np.argmax(q_table[x][y])
            obv, reward, is_done, _ = env.step(action)
            next_state = obv[0:2]

            optimal_policy[(x, y)] = action

            state = next_state

    return optimal_policy


def learn_policy(used_env):
    """
    Main function that trains the agent. Through iteration, updates the q_table and uses output_dict to output the
    optimal policy.

    :param used_env: {class instance}; the current environment.
    :return: optimal_policy {dict}; key=state:value=optimal action.
    """

    # Initializing the q-table
    x_dims = used_env.observation_space.spaces[0].n
    y_dims = used_env.observation_space.spaces[1].n
    maze_size = tuple((x_dims, y_dims))
    # Number of discrete actions
    num_actions = used_env.action_space.n
    q_table = np.zeros([maze_size[0], maze_size[1], num_actions], dtype=float)

    # Setting up the values for various factors.
    learning_factor = 0.5
    discount_factor = 0.4
    epsilon = 0.5

    episodes = 100

    for ep in range(100):

        obv = used_env.reset()
        state = obv[0:2]  # Initial state.
        is_done = False  # Terminal state status.
        MAX_T = 1000  # max trials (for one episode)
        t = 0

        while not is_done and t < MAX_T:
            t += 1

            # Using the epsilon-greedy algorithm to balance exploration and exploitation.
            if random.uniform(0, 1) > epsilon:
                action = used_env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state[0]][state[1]])  # Exploit

            # Performing the action and storing the results.
            obv, reward, is_done, _ = used_env.step(action)
            next_state = obv[0:2]

            old_value = q_table[state[0]][state[1]][action]  # Keeping a record of previous state's value to update.
            next_state_value = max(q_table[next_state[0]][next_state[1]])

            temporal_difference = reward + (discount_factor * (next_state_value - old_value))

            better_value = old_value + (learning_factor * temporal_difference)  # Updating the previous value.
            q_table[state[0]][state[1]][
                action] = better_value  # Updating the q-table with the new value for the previous state.

            state = next_state  # The actual current state as the new current state.

    optimal_policy = output_dict(used_env, q_table)
    return optimal_policy