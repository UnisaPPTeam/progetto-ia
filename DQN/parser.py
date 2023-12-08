import re
import matplotlib.pyplot as plt

DATA = re.compile("^\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*$")
FRAME_DELIMITER = re.compile("^\s*-+\s*$")


def read_file(filename: str):
    with open(filename, "r") as fp:
        x = fp.read()
    in_frame = False
    frames = list()
    frame = None
    for line in x.splitlines(False):
        if FRAME_DELIMITER.match(line):
            in_frame = not in_frame
            if in_frame:
                frame = dict()
            else:
                frames.append(frame)
        elif DATA.match(line):
            try:
                frame[DATA.match(line)[1]] = int(DATA.match(line)[2])
            except ValueError:
                frame[DATA.match(line)[1]] = float(DATA.match(line)[2])
    for key in frames[0].keys():
        feature = [(x["total_timesteps"], x[key]) for x in frames if key in x]
        timesteps, feature = zip(*feature)
        plt.plot(timesteps, feature, marker='o', linestyle='-')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(key)
        plt.show()


if __name__ == '__main__':
    read_file("/home/dar9586/file_output.txt")
