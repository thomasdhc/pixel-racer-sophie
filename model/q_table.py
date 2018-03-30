import os
import fire
import numpy as np

from environment.track_world import TrackEnv
from model import util

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = ROOT_DIR + '/../model_results'

ENV_WIDTH = 11
ENV_HEIGHT = 11

# Learning parameters
LEARNING_RATE = 0.8
FTR_DISCOUNT_Y = 0.95
NUM_EPISODES = 500

def train_q_table(env):
    q_table = np.zeros([121, 4])

    reward_list = []
    action_list = []

    for i in range(NUM_EPISODES):

        state_map = env.reset()
        state = util.reshape_state(state_map, ENV_WIDTH, 'val')
        reward_all = 0
        action_all = []
        done = False
        step = 0

        # Q-Table learning algorithm
        while step < 100:
            step += 1
            #Choose action greedily picking from Q table
            action = np.argmax(q_table[state, :] + np.random.randn(1, 4) * (1./(i + 1)))

            state_map1, reward, done = env.tick(action)
            state1 = util.reshape_state(state_map1, ENV_WIDTH, 'val')
            #Update Q-Table
            #Q-learning Q(s,a) = r + y * max_a'(Q(s',a'))
            q_table[state, action] = q_table[state, action] + \
                    LEARNING_RATE * (reward + FTR_DISCOUNT_Y * np.max(q_table[state1, :]) - q_table[state, action])

            reward_all += reward
            action_all.append(action)
            state = state1

            if done:
                print("Done racing for " + str(i))
                break

            if step == 99:
                print("For " + str(i) + " could not reach goal")

        if i == 0:
            action_list.append(action_all)
        elif i == 54:
            action_list.append(action_all)
        elif i == 499:
            action_list.append(action_all)

        reward_list.append(reward_all)
    return reward_list, action_list, NUM_EPISODES, q_table


def run():
    env = TrackEnv(ENV_WIDTH, ENV_HEIGHT)
    reward_list, action_list, num_episodes, q_table = train_q_table(env)
    count = 0
    for l in action_list:
        np.savetxt(RESULT_PATH + '/q_table_actions_{}.txt'.format(str(count)), l, fmt='%d')
        count += 1

    print("Score over time: " + str(sum(reward_list)/num_episodes))
    print("Final Q-Table Values")
    print(q_table)


if __name__ == '__main__':
    fire.Fire()



