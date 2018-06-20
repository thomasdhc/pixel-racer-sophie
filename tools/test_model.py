import os
import fire

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf

import model.util as util
from environment.track_env import TrackEnv
from tools.model_tools import load_graph

RESHAPE_TYPE = 'array'
ENV_WIDTH = 11

ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../"

def run_ckpt(model_dir, meta_graph):
    # Restoring Graph
    # This function returns a Saver
    saver = tf.train.import_meta_graph(model_dir + '/' + meta_graph)

    # We can access the default graph where all our metadata has been loaded
    graph = tf.get_default_graph()
    for op in graph.get_operations():
        print(op.name)

    # Retrieve tensors, operations, collections, etc.
    x = graph.get_tensor_by_name('input:0')
    weights = graph.get_tensor_by_name('weights:0')
    y_weight = graph.get_tensor_by_name('matmul_output:0')
    y = graph.get_tensor_by_name('output:0')

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        step = 0
        action_list = []
        track_env = TrackEnv(11, 11)
        state_map = track_env.reset()

        w = sess.run(weights)
        print(w)
        while step < 22:

            state = util.reshape_state(state_map, ENV_WIDTH, RESHAPE_TYPE)
            y_out, y_arr = sess.run([y, y_weight], feed_dict={x:state})
            print(y_arr)
            y_out = sess.run(y, feed_dict={x:state})
            action_list.append(y_out[0])
            state_map, _, done = track_env.tick(y_out[0])
            if done: break
            step += 1

        print(action_list)

def run_frozen_model(file_path):

    graph = load_graph(file_path)

    for op in graph.get_operations():
        print(op.name)

    x = graph.get_tensor_by_name('prefix/input:0')
    y = graph.get_tensor_by_name('prefix/output:0')

    with tf.Session(graph=graph) as sess:
        step = 0
        action_list = []
        track_env = TrackEnv(11, 11)
        state_map = track_env.reset()

        while step < 22:

            state = util.reshape_state(state_map, ENV_WIDTH, RESHAPE_TYPE)
            y_out = sess.run(y, feed_dict={x:state})
            action_list.append(y_out[0])
            state_map, _, done = track_env.tick(y_out[0])
            if done:
                break
            step += 1
        print(action_list)

if __name__ == '__main__':
    fire.Fire({
        'run_frozen': run_frozen_model,
        'run_ckpt': run_ckpt
    })