import numpy as np
from shapely.geometry import LineString, Polygon
import cv2


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


class SegMask():
    def __init__(self, points: np.ndarray = None, label: int = None):
        self.points = points
        self.label = label
    

    def from_lane_line(self, line: LaneLine, shape=(1920,1080)):
        self.points = self.from_line_to_mask(line, shape, shape=shape)
    

    @staticmethod
    def from_line_to_mask(line: LaneLine, shape=(1920, 1080), tolerance=0.0015):
        if line.points.shape[0] <= 0:
            return None

        # shape = (shape[1], shape[0])

        image = np.zeros((shape[1], shape[0], 1), dtype=np.uint8)
        
        if line.points.shape[0] == 1:
            cv2.circle(image, (line.points[0] * np.array(shape)).astype(int), 10, (255), -1) # You may need to adjust the radius
        else:
            for idx in range(1, len(line.points)):
                cv2.line(image, 
                        (line.points[idx - 1] * np.array(shape)).astype(int),
                        (line.points[idx] * np.array(shape)).astype(int), 
                        color=(255), 
                        thickness=20) # You may need to adjust the thickness
        
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        points = np.array(contours[0]).reshape(-1, 2).astype(np.float32)
        points /= np.array(shape).astype(np.float32)

        simplified_polygon = Polygon(points).simplify(tolerance, preserve_topology=True)
        points = np.array(simplified_polygon.exterior.coords)

        return points



def simplify_lines(lines: LaneLine, tolerance=0.008):
    for line in lines:
        simplify_line(line, tolerance)