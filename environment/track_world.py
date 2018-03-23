import time, os

import fire
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import environment.entity as e
from environment.entity import Coord

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = ROOT_DIR + '/../model_results'

class Track_Env():
	def __init__(self, size_x, size_y):
		self.size_x = size_x
		self.size_y = size_y
		self.env = np.zeros([self.size_y, self.size_x])
		self.reset()


	def reset(self):
		#List of environment objects
		self.walls = []
		self.objs = []

		self.walls.append(e.Line_Wall(Coord(0, 10), Coord(6, 10)))
		self.walls.append(e.Line_Wall(Coord(0, 7), Coord(6, 7)))
		self.walls.append(e.Line_Wall(Coord(0, 7), Coord(0, 10))) 
		self.walls.append(e.Line_Wall(Coord(3, 3), Coord(5, 3)))
		self.walls.append(e.Line_Wall(Coord(3, 0), Coord(5, 0)))
		self.walls.append(e.Line_Wall(Coord(3, 0), Coord(3, 3)))
		self.walls.append(e.Half_Circle_Wall(Coord(5, 5), 5, 'e'))
		self.walls.append(e.Half_Circle_Wall(Coord(5, 5), 2, 'e'))

		self.objs.append(e.Race_Car(Coord(1, 9)))

		return self.render_env()


	#In array creation the x and y must be flipped to reflect in the graph
	def render_env(self):
		self.env = np.zeros([self.size_y, self.size_x])
		for wall in self.walls:
			for location in wall.get_wall_pixel_locations():
				self.env[location.y, location.x] = 1

		#Goal
		self.env[1, 4] = 0.25
		self.env[2, 4] = 0.25
		
		for obj in self.objs:
			self.env[obj.y, obj.x] = 0.5

		return self.env

	# '0' = down, '1' = left, '2' = up, '3' = right
	def move_obj(self, action):
		penalty = 0
		racer = self.objs[0]
		offset_x, offset_y = 0, 0

		if action == 0:
			offset_y = 1
		elif action == 1:
			offset_x = -1
		elif action == 2:
			offset_y = -1
		elif action == 3:
			offset_x = 1

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


def run(filename):

	fig = plt.figure()

	action_list = np.loadtxt(RESULT_PATH + '/' + filename, dtype=int)
	run.num_action = 0
	track_env = Track_Env(11, 11)
	env = track_env.reset()
	im = plt.imshow(env, animated='True')
	
	def update_fig(self):
		if run.num_action < len(action_list):
			new_env, _, _ = track_env.tick(action_list[run.num_action])
			im.set_array(new_env)
			run.num_action += 1
		return im,

	ani = animation.FuncAnimation(fig, update_fig, frames=len(action_list) - 1, interval=150, blit=True)
	plt.show()

if __name__ == '__main__':
	fire.Fire({
		'run': run
	})
