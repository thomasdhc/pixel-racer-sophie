import fire, os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from environment.track_env import TrackEnv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = ROOT_DIR + '/../model_action_output'

def loss_graph(filename):
    loss = np.loadtxt(RESULT_PATH + '/' + filename, dtype=float)
    plt.plot(loss)
    plt.show()


def track(track_name):
    track_env = TrackEnv(11, 11, track_name)
    env = track_env.reset()
    _ = plt.imshow(env, origin='lower')
    plt.show()


def generate_animation(filename):
    fig = plt.figure()
    action_list = np.loadtxt(RESULT_PATH + '/' + filename, dtype=int)
    save_gif.num_action = 0
    track_env = TrackEnv(11, 11, "track1")
    env = track_env.reset()
    im = plt.imshow(env, animated='True', origin='lower')

    def update_fig(self):
        if save_gif.num_action == 0:

            new_env = track_env.reset()
            im.set_array(new_env)
        else:
            new_env, _, _ = track_env.tick(action_list[save_gif.num_action - 2])
            im.set_array(new_env)
        save_gif.num_action += 1
        return im,

    return animation.FuncAnimation(fig, update_fig, frames=len(action_list) + 1, interval=150, blit=False, repeat=False)


def save_gif(filename):
    ani = generate_animation(filename)
    ani.save(ROOT_DIR + '/../gifs/' + filename[:-4] + '.gif', writer='imagemagick', fps=7)


def model_action(filename):
    _ = generate_animation(filename)
    plt.show()


if __name__ == '__main__':
    fire.Fire({
        'loss': loss_graph,
        'gif' : save_gif,
        'track' : track,
        'action' : model_action
    })