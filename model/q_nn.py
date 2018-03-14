import os, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from environment.track_world import Track_Env
from model import util

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = ROOT_DIR + '/../model_results'

ENV_WIDTH = 11
ENV_HEIGHT = 11
env = Track_Env(ENV_WIDTH, ENV_HEIGHT)

lr = 0.1

tf.reset_default_graph()

# Feed-forward network
in_var = tf.placeholder(shape=[1,121], dtype=tf.float32)
weights = tf.Variable(tf.random_uniform([121,4], 0, 0.01))
q_out = tf.matmul(in_var, weights)
predict = tf.argmax(q_out, 1)

q_next = tf.placeholder(shape=[1,4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(q_next - q_out))
trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
update_model = trainer.minimize(loss)

init = tf.global_variables_initializer()

# Learning parameters 
y = 0.99
e = 0.1
num_episodes = 1000

action_list = []
reward_list = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
    
        state_map = env.reset()
        state = util.reshape_state(state_map, ENV_WIDTH)
        reward_all = 0
        action_all = []
        done = False
        step = 0

        while step < 99:
            step += 1
            action, all_q = sess.run([predict, q_out], feed_dict={in_var:np.identity(121)[state:state + 1]})

            if np.random.rand(1) < e:
                action[0] = np.random.randint(0,3)

            state_map1, reward, done = env.tick(action[0])

            state1 = util.reshape_state(state_map1, ENV_WIDTH)
            q1 = sess.run(q_out, feed_dict={in_var:np.identity(121)[state1:state1+1]})

            max_q1 = np.max(q1)
            target_q = all_q
            target_q[0, action[0]] = reward + y * max_q1

            #Train our network using target and predicted Q values
            _, weights1 = sess.run([update_model, weights], feed_dict={in_var:np.identity(121)[state:state + 1], q_next:target_q})
            reward_all += reward
            action_all.append(action)
            state = state1

            if done == True:
                e = 1./((i/50) + 10)
                print ("Done racing for " + str(i))
                break

            if step == 98:
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
    np.savetxt(RESULT_PATH + '/q_nn_actions_{}.txt'.format(str(count)), l, fmt='%d')
    count += 1
print ("Percent of sucessful episodes: " + str(sum(reward_list)/num_episodes)+ "%")

        
