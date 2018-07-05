import os
import environment.entity as e

from environment.entity import Coord

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_track_from_file(track_name):
    track_file = open(ROOT_DIR + "/generated-track/" + track_name + ".txt", "r")
    count = int(track_file.readline())
    walls = []
    objs = []
    goal_coord = []

    for _ in range(0, count):
        arr = track_file.readline().split()

        if arr[0] == "lw":
            walls.append(e.Line_Wall(Coord(int(arr[1]),int(arr[2])), Coord(int(arr[3]), int(arr[4]))))
        elif arr[0] == "hc":
            walls.append(e.Half_Circle_Wall(Coord(int(arr[1]), int(arr[2])), int(arr[3]), arr[4]))
        elif arr[0] == "r":
            objs.append(e.Race_Car(Coord(int(arr[1]), int(arr[2]))))
        elif arr[0] == "g":
            for i in range(0, (len(arr) -  1) // 2):
                goal_coord.append(Coord(int(arr[i * 2 + 1]), int(arr[i * 2 + 2])))

    goal = e.Goal(goal_coord)

    return walls, objs, goal