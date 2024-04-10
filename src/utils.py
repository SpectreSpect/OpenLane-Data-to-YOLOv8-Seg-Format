import numpy as np
from shapely.geometry import LineString


class LaneLine():
    def __init__(self, points: np.ndarray, label: int):
        self.points = points
        self.label = label


def simplify_line(line: LaneLine):
    line_string = LineString(line.points)
    simplified_line_string = line_string.simplify(0.008)
    line.points = np.array(list(simplified_line_string.coords))


def simplify_lines(lines: LaneLine):
    for line in lines:
        simplify_line(line)