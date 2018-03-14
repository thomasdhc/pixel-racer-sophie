import os 
import numpy as np

from environment.track_world import Track_Env
from model import util 

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = ROOT_DIR + '/../model_results'

ENV_WIDTH = 11
ENV_HEIGHT = 11
env = Track_Env(ENV_WIDTH, ENV_HEIGHT)
q_table = np.zeros([121, 4])

# Learning parameters
learning_rate = 0.8
ftr_discount_y = 0.95
num_episodes = 500

reward_list = []
action_list = []

for i in range(num_episodes):

    state_map = env.reset()
    state = util.reshape_state(state_map, ENV_WIDTH)
    reward_all = 0
    action_all = []
    done = False
    step = 0

    # Q-Table learning algorithm
    while step < 100:
        step += 1
        #Choose action greedily picking from Q table
        action = np.argmax(q_table[state,:] + np.random.randn(1, 4) * (1./(i + 1)))

        state_map1, reward, done = env.tick(action)
        state1 = util.reshape_state(state_map1, ENV_WIDTH)
        #Update Q-Table
        #Q-learning Q(s,a) = r + y * max_a'(Q(s',a')) 
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + ftr_discount_y * np.max(q_table[state1,:]) - q_table[state,action])

        reward_all += reward
        action_all.append(action) 
        state = state1

        if done == True:
            print ("Done racing for " + str(i))
            break

        if step == 99:
            print ("For " + str(i) + " could not reach goal")

    if i == 0:
        action_list.append(action_all)
    elif i == 54:
        action_list.append(action_all)
    elif i == 499:
        action_list.append(action_all)

    reward_list.append(reward_all)

count = 0
for l in action_list:
    np.savetxt(RESULT_PATH + '/q_table_actions_{}.txt'.format(str(count)), l, fmt='%d')
    count += 1

print ("Score over time: " + str(sum(reward_list)/num_episodes))
print ("Final Q-Table Values")
print (q_table)



