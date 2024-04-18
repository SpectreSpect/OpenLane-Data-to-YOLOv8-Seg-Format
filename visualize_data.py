import cv2
import numpy as np
import random
from shapely.geometry import LineString, Polygon
import src.utils as utils


category_dict = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    20: 13,
    21: 14
    }


def draw_curve(image, points, color=(0, 0, 255), thickness=2, radius=6):
    width = image.shape[1]
    height = image.shape[0]
    for idx in range(1, len(points)):
        pt1x = int(points[idx - 1][0] * float(width))
        pt1y = int(points[idx - 1][1] * float(height))
        pt2x = int(points[idx][0] * float(width))
        pt2y = int(points[idx][1] * float(height))
        
        cv2.line(image, (pt1x, pt1y), (pt2x, pt2y), color, thickness)
    
        cv2.circle(image, (pt1x, pt1y), radius, color, -1)
        if idx >= len(points) - 1:
            cv2.circle(image, (pt2x, pt2y), radius, color, -1)


def get_lane_lines(path):
    lane_lines = []
    with open(path, "r") as file:
        for line_idx, line in enumerate(file.readlines()):
            split_line = line.split(' ')
            category = int(split_line[0])
            if split_line[-1] == '\n':
                coords = split_line[1:-1]
            else:
                coords = split_line[1:]
            
            if len(coords) % 2 != 0:
                print(f"Invalid data - line_idx: {line_idx}  category: {category}  len(coords): {len(coords)}")
                return None
            
            nxy = []
            for idx in range(0, len(coords), 2):
                nx = float(coords[idx])
                ny = float(coords[idx + 1])
                nxy.append([nx, ny])
            
            lane_line = utils.LaneLine(np.array(nxy), category)
            lane_lines.append(lane_line)
    return lane_lines


def get_pose_lane_lines(path: str):
    lane_lines = []
    bounding_boxes = []
    with open(path, "r") as file:
        for line_idx, line in enumerate(file.readlines()):
            split_line = line.split(' ')
            category = int(split_line[0])

            bounding_box = dict()
            bounding_box["label"] = category
            bounding_box["u"] = float(split_line[1])
            bounding_box["v"] = float(split_line[2])
            bounding_box["width"] = float(split_line[3])
            bounding_box["height"] = float(split_line[4])

            bounding_boxes.append(bounding_box)

            if split_line[-1] == '\n':
                coords = split_line[5:-1]
            else:
                coords = split_line[5:]
            
            if len(coords) % 2 != 0:
                print(f"Invalid data - line_idx: {line_idx}  category: {category}  len(coords): {len(coords)}")
                return None
            
            nxy = []
            for idx in range(0, len(coords), 2):
                nx = float(coords[idx])
                ny = float(coords[idx + 1])
                nxy.append([nx, ny])
            
            lane_line = utils.LaneLine(np.array(nxy), category)
            lane_lines.append(lane_line)
    return lane_lines, bounding_boxes


def draw_lane_lines(image, lines, palette):
    for line in lines:
        label = line.label
        points = line.points
        draw_curve(image, points, palette[label], radius=5)


def draw_bounding_box(image, bounding_box, color=(0, 0, 255), thickness=2):
    top_left_u = bounding_box["u"] - bounding_box["width"] / 2.0
    top_left_v = bounding_box["v"] - bounding_box["height"] / 2.0

    bottom_right_u = bounding_box["u"] + bounding_box["width"] / 2.0
    bottom_right_v = bounding_box["v"] + bounding_box["height"] / 2.0

    top_left = [int(top_left_u * image.shape[1]), int(top_left_v * image.shape[0])]
    bottom_right = [int(bottom_right_u * image.shape[1]), int(bottom_right_v * image.shape[0])]

    cv2.rectangle(image, top_left, bottom_right, color, thickness)


def draw_bounding_boxes(image, bounding_boxes, palette):
    for bounding_box in bounding_boxes:
        category = bounding_box["label"]
        draw_bounding_box(image, bounding_box, palette[category])
        # draw_curve(image, points, palette[category], radius=5)


def generate_palette(values_dict: dict):
    palette = {}
    for key, value in values_dict.items():
        palette[value] = [random.random() * 255.0, random.random() * 255.0, random.random() * 255.0]
    return palette


def draw_mask_from_lines(image, lines, palette, thickness=-1):
    for line in lines:
        counter = (line.points * np.array([image.shape[1], image.shape[0]])).astype(int)
        cv2.drawContours(image, [counter], -1, palette[line.label], thickness)


def simplify_counter(line: utils.LaneLine, tolerance=0.008):
    line.points = Polygon(line.points).simplify(tolerance, preserve_topology=True)
    line.points = np.array(line.points.exterior.coords)


def simplify_counters(lines, tolerance=0.008):
    for line in lines:
        simplify_counter(line, tolerance)


if __name__ == "__main__":
    image = cv2.imread("data/yolov8_medium-500-masks/images/train/151163701004979800.jpg")
    lines, bounding_boxes = get_pose_lane_lines("data/yolov8_medium-500-masks/labels/train/151163701004979800.txt")
    palette = generate_palette(category_dict)
    draw_mask_from_lines(image, lines, palette)
    # palette = generate_palette(category_dict)
    # draw_lane_lines(image, lines, palette)

    cv2.imshow('Image', cv2.resize(image, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()