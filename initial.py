import gym
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1
DISCOUNT = 0.92
EPISODES = 25000
SHOW_EVERY = 1000

#load model
env = gym.make("MountainCar-v0")

DESCRETE_OS_SIZE = [36] * len(env.observation_space.high)
descrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DESCRETE_OS_SIZE

epsilon = 0.4
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYION = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYION - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DESCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_descrete_state(state):
    descrete_state = (state - env.observation_space.low) / descrete_os_win_size
    return tuple(descrete_state.astype(np.int))

for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    descrete_state = get_descrete_state(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[descrete_state])
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_descrete_state = get_descrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_descrete_state])
            current_q = q_table[descrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[descrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[descrete_state + (action, )] = 0

        descrete_state = new_descrete_state
    
    if END_EPSILON_DECAYION >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    ep_rewards.append(episode_reward)
    if not episode % SHOW_EVERY:
        np.save(f"qtables/{episode}-qtable.npy", q_table)
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Ep: {episode} Avg: {average_reward} Min: {min(ep_rewards[-SHOW_EVERY:])} Max: {max(ep_rewards[-SHOW_EVERY:])}")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()

