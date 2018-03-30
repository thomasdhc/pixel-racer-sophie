import os
import fire
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from environment.track_world import TrackEnv
from model import util

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = ROOT_DIR + '/../model_results'

ENV_WIDTH = 11
ENV_HEIGHT = 11

# Feed-forward network
class FeedForwardNetwork():
    def __init__(self, input_size, out_size, lr):
        self.in_var = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
        self.weights = tf.Variable(tf.random_uniform([input_size, out_size], 0, 0.01))
        self.q_out = tf.matmul(self.in_var, self.weights)
        self.predict = tf.argmax(self.q_out, 1)

        self.q_next = tf.placeholder(shape=[1, out_size], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.q_next - self.q_out))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update_model = self.trainer.minimize(self.loss)


def train_model(env, reshape_type, lr):
    tf.reset_default_graph()
    ffn = FeedForwardNetwork(121, 4, lr)
    init = tf.global_variables_initializer()

    # Learning parameters
    y = 0.99
    e = 0.1
    num_episodes = 1000

    action_list = []
    reward_list = []
    loss_list = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):

            state_map = env.reset()
            state = util.reshape_state(state_map, ENV_WIDTH, reshape_type)
            reward_all = 0.
            action_all = []
            loss_total = 0.
            done = False
            step = 0

            while step < 99:
                step += 1

                action, all_q = sess.run([ffn.predict, ffn.q_out], feed_dict={ffn.in_var: state})

                if np.random.rand(1) < e:
                    action[0] = np.random.randint(0, 4)

                state_map1, reward, done = env.tick(action[0])
                state1 = util.reshape_state(state_map1, ENV_WIDTH, reshape_type)
                q1 = sess.run(ffn.q_out, feed_dict={ffn.in_var:state1})

                max_q1 = np.max(q1)
                target_q = all_q
                target_q[0, action[0]] = reward + y * max_q1

                # Train our network using target and predicted Q values
                loss, _ = sess.run([ffn.loss, ffn.update_model], feed_dict={ffn.in_var: state, ffn.q_next:target_q})
                loss_total += loss
                reward_all += reward
                action_all.append(action)
                state = state1

                if done:
                    e = 1./((i/50) + 10)
                    print("Done racing for " + str(i))
                    break

                if step == 98:
                    print("For " + str(i) + " could not reach goal")

            if i % 2 == 0:
                action_list.append(action_all)
            reward_list.append(reward_all)
            loss_list.append(loss_total)
        test(sess, ffn, reshape_type)

    return action_list, reward_list, loss_list, num_episodes


def test(sess, ffn, reshape_type):
    test_env = TrackEnv(ENV_WIDTH, ENV_HEIGHT)
    state_org = test_env.reset()
    test_actions = [2, 3, 0, 3, 2, 3, 0, 3, 2, 3, 0, 3, 2, 2, 3, 3, 2, 1, 2, 3, 2, 1, 2, 1, 2, 2, 1, 0, 1, 2, 1]
    print(state_org)
    print('\n')
    for action in test_actions:
        state_map, _, _ = test_env.tick(action)
        state = util.reshape_state(state_map, ENV_WIDTH, reshape_type)
        act, all_q = sess.run([ffn.predict, ffn.q_out], feed_dict={ffn.in_var: state})
        print(action)
        print(state_map)
        print(all_q)
        print(act)
        print('\n')


def run():
    env = TrackEnv(ENV_WIDTH, ENV_HEIGHT)
    a_list, r_list, l_list, num_ep = train_model(env, 'identity', 0.1)
    count = 0
    for action_array in a_list:
        np.savetxt(RESULT_PATH + '/q_nn_actions_{}.txt'.format(str(count)), action_array, fmt='%d')
        count += 1

    success = 0
    for r in r_list:
        if r == 1.:
            success += 1
    print("Percent of successful episodes: " + str(success/num_ep * 100)+ "%")
    plt.plot(l_list)
    plt.show()


if __name__ == '__main__':
    fire.Fire()
