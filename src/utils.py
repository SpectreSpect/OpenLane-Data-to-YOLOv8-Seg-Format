import numpy as np
from shapely.geometry import LineString


class LaneLine():
    def __init__(self, points: np.ndarray, label: int):
        self.points = points
        self.label = label


def simplify_line(line: LaneLine, tolerance=0.008, verbose=0):
    if line.points.shape[0] > 1:
        if verbose == 1:
            print(f"{line.points.shape[0]} -> ", end="")

        line_string = LineString(line.points)
        simplified_line_string = line_string.simplify(tolerance)
        line.points = np.array(list(simplified_line_string.coords))

        if verbose == 1:
            print(f"{line.points.shape[0]}")


def simplify_lines(lines: LaneLine, tolerance=0.008):
    for line in lines:
        simplify_line(line, tolerance)