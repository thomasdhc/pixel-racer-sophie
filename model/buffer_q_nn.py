import os
import fire
import numpy as np
import tensorflow as tf

from environment.track_env import TrackEnv
from model.experience_buffer import ExperienceBuffer
from model import util

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = ROOT_DIR + '/../model_action_output'
MODEL_PATH = ROOT_DIR + '/../model_results_buff_qnn'

ENV_WIDTH = 11
ENV_HEIGHT = 11

# Feed-forward network
class FeedForwardNetwork():
    def __init__(self, input_size, out_size, lr):
        self.in_var = tf.placeholder(shape=[1, input_size], dtype=tf.float32, name="input")
        self.weights = tf.Variable(tf.random_uniform([input_size, out_size], 0, 0.01), name="w1")
        self.q_out = tf.matmul(self.in_var, self.weights, name="output_w")
        self.predict = tf.argmax(self.q_out, 1, name="output")

        self.q_next = tf.placeholder(shape=[1, out_size], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.q_next - self.q_out))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update_model = self.trainer.minimize(self.loss)


# Learning parameters
DISCOUNT_FACTOR = 0.99
NUM_EP = 10000
ANNEALING_STEP = 10000.
START_E = 1.
END_E = 0.1
RDC_E = (START_E - END_E) / ANNEALING_STEP

PRE_TRAIN_STEPS = 10000
BUFFER_SIZE = 5000

RESHAPE_TYPE = 'array'

LEARNING_RATE = 0.01
# Rate to update target network toward primary network
TAU = 0.1


def update_target_graph(tf_vars):
    num_vars = len(tf_vars)
    op_holder = []
    for idx, var in enumerate(tf_vars[0:num_vars//2]):
        op_holder.append(tf_vars[idx + num_vars//2].assign( \
            (TAU * var.value()) + ((1 - TAU) * tf_vars[idx + num_vars//2].value())))
    return op_holder


def generate_episode(e, total_steps, env, sess, main_ffn, state):
    if np.random.rand(1) < e or total_steps < PRE_TRAIN_STEPS:
        action = [np.random.randint(0, 4)]
    else:
        action = sess.run(main_ffn.predict, feed_dict={main_ffn.in_var: state})

    state_map1, reward, done = env.tick(action[0])
    state1 = util.reshape_state(state_map1, ENV_WIDTH, RESHAPE_TYPE)

    return np.reshape(np.array([state, action, reward, state1, done]), [1, 5]), state1, done


def update_model(train_buffer, sess, main_ffn, target_ffn):
    train_batch = train_buffer.sample(1)[0]

    main_q = sess.run(main_ffn.q_out, feed_dict={main_ffn.in_var:train_batch[0]})
    future_q = sess.run(target_ffn.q_out, feed_dict={target_ffn.in_var:train_batch[3]})

    max_q1 = np.max(future_q)
    target_q = main_q

    target_q[0, train_batch[1]] = train_batch[2] + DISCOUNT_FACTOR * max_q1

    loss, _ = sess.run([main_ffn.loss, main_ffn.update_model], \
                       feed_dict={main_ffn.in_var: train_batch[0], main_ffn.q_next:target_q})

    return loss

def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)


def train_model(env):
    tf.reset_default_graph()

    e = START_E
    total_steps = 0

    main_ffn = FeedForwardNetwork(121, 4, LEARNING_RATE)
    target_ffn = FeedForwardNetwork(121, 4, LEARNING_RATE)

    train_var = tf.trainable_variables()
    target_ops = update_target_graph(train_var)

    saver = tf.train.Saver()

    train_buffer = ExperienceBuffer(BUFFER_SIZE)

    loss_list = []

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        episode_buffer = ExperienceBuffer(BUFFER_SIZE)

        for i in range(NUM_EP):
            state_map = env.reset()
            state = util.reshape_state(state_map, ENV_WIDTH, RESHAPE_TYPE)

            loss_total = 0.
            step = 0

            while step < 99:
                step += 1

                ep_list, state1, done = generate_episode(e, total_steps, env, sess, main_ffn, state)
                total_steps += 1
                episode_buffer.add(ep_list)

                if total_steps > PRE_TRAIN_STEPS:
                    if e > END_E:
                        e -= RDC_E

                    loss = update_model(train_buffer, sess, main_ffn, target_ffn)
                    update_target(target_ops, sess)
                    loss_total += loss

                state = state1
                if done:
                    break

            train_buffer.add(episode_buffer.buffer)
            loss_list.append(loss_total)
            if i % 500 == 0:
                print(f"Training step {str(i)} complete!")
        test(sess, main_ffn)
        store_final_nn_actions(sess, main_ffn)
        saver.save(sess, MODEL_PATH + '/buffer-qnn-model-' + str(NUM_EP) + '.ckpt')
    return loss_list


def test(sess, ffn):
    step = 0
    action_list = []
    track_env = TrackEnv(11, 11)
    state_map = track_env.reset()

    while step < 22:

        state = util.reshape_state(state_map, ENV_WIDTH, RESHAPE_TYPE)
        act, all_q = sess.run([ffn.predict, ffn.q_out], feed_dict={ffn.in_var:state})
        print(all_q)
        action_list.append(act[0])
        state_map, _, done = track_env.tick(act[0])
        if done:
            break
        step += 1
    print(action_list)


def store_final_nn_actions(sess, ffn):
    test_env = TrackEnv(ENV_WIDTH, ENV_HEIGHT)
    state_map = test_env.reset()
    state = util.reshape_state(state_map, ENV_WIDTH, RESHAPE_TYPE)
    step = 0
    action_array = []
    while step < 99:
        step += 1
        action = sess.run(ffn.predict, feed_dict={ffn.in_var: state})
        action_array.append(action)
        state_map, _, done = test_env.tick(action[0])
        state = util.reshape_state(state_map, ENV_WIDTH, RESHAPE_TYPE)
        if done:
            break

    np.savetxt(RESULT_PATH + '/buffer_q_nn.txt', action_array, fmt='%d')


def run():
    env = TrackEnv(ENV_WIDTH, ENV_HEIGHT)
    l_list = train_model(env)
    np.savetxt(RESULT_PATH + '/buffer_q_nn_loss.txt', l_list, fmt='%f')


if __name__ == '__main__':
    fire.Fire()
