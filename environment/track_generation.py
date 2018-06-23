import os

import fire
from os import listdir, path

UNIQUE_TRACK_DIR = path.dirname(path.abspath(__file__)) + "/unique-track"
GENERATED_TRACK_DIR = path.dirname(path.abspath(__file__)) + "/generated-track"

def generate_new_tracks():

    unique_track_files = [f for f in listdir(UNIQUE_TRACK_DIR) if path.isfile(path.join(UNIQUE_TRACK_DIR, f))]
    for idx, file in enumerate(unique_track_files):
        track_file = open(UNIQUE_TRACK_DIR + "/" + file, "r")
        count = int(track_file.readline())

        # Vertical Flip
        vf_track = open(GENERATED_TRACK_DIR + f"/track{str(idx)}_1", "w")
        # Horizontal Flip
        hf_track = open(GENERATED_TRACK_DIR + f"/track{str(idx)}_2", "w")
        # 90 Degree Rotations
        ro_track = [ open(GENERATED_TRACK_DIR + f"/track{str(idx)}_{str(i)}", "w") for i in range(3, 6) ]

        vf_track.write("10\n")
        hf_track.write("10\n")
        [ new_track.write("10\n") for new_track in ro_track ]

        for _ in range(0, count):
            arr = track_file.readline().split()

            if arr[0] == "lw":
                x1, y1, x2, y2 = (int(arr[1]), int(arr[2]), int(arr[3]), int(arr[4]))

                vf_track.write("lw %d %d %d %d\n" % (5 + 5 - x1, y1, 5 + 5 - x2, y2))
                hf_track.write("lw %d %d %d %d\n" % (x1, 5 + 5 - y1, x2, 5 + 5 - y2))

                for i in range(0, 3):
                    x1_tmp = x1
                    x1 = y1
                    y1 = 10 - x1_tmp

                    x2_tmp = x2
                    x2 = y2
                    y2 = 10 - x2_tmp

                    ro_track[i].write("lw %d %d %d %d\n" % (x1, y1, x2, y2))

            elif arr[0] == "hc":
                x, y = (int(arr[1]), int(arr[2]))
                card_dir = arr[4]

                v_flip = {"e": "w", "w": "e", "n": "n", "s": "s"}
                h_flip = {"e": "e", "w": "w", "n": "s", "s": "n"}
                ro = {"e": "s", "s": "w", "w": "n", "n": "e"}

                vf_track.write("hc %d %d %s %s\n" % (5 + 5 - x, y, arr[3], v_flip[card_dir]))
                hf_track.write("hc %d %d %s %s\n" % (x, 5 + 5 - y, arr[3], h_flip[card_dir]))

                for i in range(0, 3):
                    x_tmp = x
                    x = y
                    y = 10 - x_tmp
                    card_dir = ro[card_dir]

                    ro_track[i].write("hc %d %d %s %s\n" % (x, y, arr[3], card_dir))

            elif arr[0] == "r":
                x, y = (int(arr[1]), int(arr[2]))

                vf_track.write("r %d %d\n" %(5 + 5 - x, y))
                hf_track.write("r %d %d\n" %(x, 5 + 5 - y))

                for i in range(0, 3):
                    x_tmp = x
                    x = y
                    y = 10 - x_tmp
                    ro_track[i].write("r %d %d\n" %(x, y))


            elif arr[0] == "g":
                vf_goal = "g"
                hf_goal = "g"
                ro_goal = ["g", "g", "g"]
                for i in range(0, (len(arr) - 1) // 2):
                    x, y = (int(arr[i * 2 + 1]), int(arr[i * 2 + 2]))

                    vf_goal += " " + str(5 + 5 - x) + " " + str(y)
                    hf_goal += " " + str(x) + " " + str(5 + 5 - y)

                    for j in range(0, 3):
                        x_tmp = x
                        x = y
                        y = 10 - x_tmp
                        ro_goal[j] += " " + str(x) + " " + str(y)

                vf_track.write(vf_goal)
                hf_track.write(hf_goal)
                [ new_track.write(ro_goal[idx] + "\n") for idx, new_track in enumerate(ro_track) ]

        vf_track.close()
        hf_track.close()
        [new_track.close() for new_track in ro_track]


if __name__ == '__main__':
    fire.Fire({
        'generate': generate_new_tracks
    })