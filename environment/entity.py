from enum import Enum
from typing import List

import numpy as np


class Coord():
    def __init__(self,x=None,y=None):
        self.x = x
        self.y = y


class Race_Car():
    def __init__(self, position):
        self.x = position.x
        self.y = position.y
        self.size = 1
        self.speed = 1

class Goal():
    def __init__(self, coordinates):
        self.coordinates = coordinates

class Wall_Type(Enum):
    NONE = 0
    LINE = 1
    HALF_CIRCLE = 2
    QUARTER_CIRCLE = 3


class Wall():
    def __init__(self):
        self.type = Wall_Type.NONE


class Line_Wall(Wall):
    def __init__(self, start, end):
        self.start: Coord = start
        self.end: Coord = end
        Wall.type = Wall_Type.LINE


    def get_wall_pixel_locations(self):
        pixel_locations: List[Coord] = []

        for x in range (0, abs(self.start.x - self.end.x)):
            if (self.start.x > self.end.x):
                pixel_locations.append(Coord(self.start.x - x, self.start.y))
            else:
                pixel_locations.append(Coord(self.start.x + x, self.start.y))


        for y in range (0, abs(self.start.y - self.end.y)):
            if (self.start.y > self.end.y):
                pixel_locations.append(Coord(self.start.x, self.start.y - y))
            else:
                pixel_locations.append(Coord(self.start.x, self.start.y + y))

        return pixel_locations


class Half_Circle_Wall(Wall):
    def __init__(self, center, radius, qudrant):
        self.center = center
        self.radius = radius
        self.qudrant = qudrant
        Wall.type = Wall_Type.HALF_CIRCLE


    #Mid-point circle drawing algorithm
    def get_wall_pixel_locations(self):
        pixel_locations = []

        first_quadrant = []
        second_quadrant = []
        third_quadrant = []
        fourth_quadrant = []

        r = self.radius
        x, y = r, 0

        first_quadrant.append(Coord(x + self.center.x, y + self.center.y))

        if r > 0:
            second_quadrant.append(Coord(y + self.center.x, x + self.center.y))
            third_quadrant.append(Coord(-x + self.center.x, y + self.center.y))
            fourth_quadrant.append(Coord(-y + self.center.x, -x + self.center.y))

        p = 1 - r

        while x > y:
            y += 1

            if p <= 0:
                p = p + 2 * y + 1
            else:
                x -= 1
                p = p + 2 * y - 2 * x + 1

            if x < y:
                break

            first_quadrant.append(Coord(x + self.center.x, y + self.center.y))
            second_quadrant.append(Coord(-x + self.center.x, y + self.center.y))
            third_quadrant.append(Coord(-x + self.center.x, -y + self.center.y))
            fourth_quadrant.append(Coord(x + self.center.x, -y + self.center.y))

            if x != y:
                first_quadrant.append(Coord(y + self.center.x, x + self.center.y))
                second_quadrant.append(Coord(-y + self.center.x, x + self.center.y))
                third_quadrant.append(Coord(-y + self.center.x, -x + self.center.y))
                fourth_quadrant.append(Coord(y + self.center.x, -x + self.center.y))

        if self.qudrant == 'e':
            pixel_locations = first_quadrant + fourth_quadrant
        elif self.qudrant == 'n':
            pixel_locations = first_quadrant + second_quadrant
        elif self.qudrant == 'w':
            pixel_locations = second_quadrant + third_quadrant
        elif self.qudrant == 's':
            pixel_locations = third_quadrant + fourth_quadrant

        return pixel_locations
