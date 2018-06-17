import os

import fire
import numpy as np

import environment.entity as e
import environment.helper as helper
from environment.entity import Coord

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = ROOT_DIR + '/../model_action_output'

class TrackEnv():
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.env = np.zeros([self.size_y, self.size_x])
        self.reset()

    def reset(self):
        self.walls, self.objs, self.goal = helper.generate_track_from_file()
        return self.render_env()


    # In array creation the x and y must be flipped to reflect in the graph
    def render_env(self):
        self.env = np.zeros([self.size_y, self.size_x])

        # Walls
        for wall in self.walls:
            for location in wall.get_wall_pixel_locations():
                self.env[location.y, location.x] = 1

        # Goal
        for coord in self.goal.coordinates:
            self.env[coord.y, coord.x] = 0.25

        # Objects
        for obj in self.objs:
            self.env[obj.y, obj.x] = 0.5

        return self.env

    # '0' = down, '1' = left, '2' = up, '3' = right
    def move_obj(self, action):
        penalty = 0
        racer = self.objs[0]

        offset_map = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}
        offset_x, offset_y = offset_map[action]

        # Remember to flip the y and x coordinates when indexing into 2D array
        if self.env[racer.y + offset_y, racer.x + offset_x] != 1:
            racer.x = racer.x + offset_x
            racer.y = racer.y + offset_y
        else:
            penalty = 0

        self.objs[0] = racer

        return penalty


    def check_goal(self):
        racer = self.objs[0]

        done = False
        reward = 0

        if self.env[racer.y, racer.x] == 0.25:
            done = True
            reward = 1

        return reward, done


    def tick(self, action):
        penalty = self.move_obj(action)
        reward, done = self.check_goal()
        self.render_env()

        return self.env, (reward + penalty), done