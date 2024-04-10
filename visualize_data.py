import cv2
import numpy as np
import random
from shapely.geometry import LineString
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


# def get_lane_lines(path: str) -> dict:
#     with open(path, "r") as file:
#         for line in file.readlines():


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
            
            # lane_line = {"nxy": np.array(nxy), "category": category}
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
            
            # lane_line = {"nxy": np.array(nxy), "category": category}
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

    # cv2.circle(image, top_left, 10, (0, 255, 0), thickness=-1)
    # cv2.circle(image, top_left, 10, (0, 255, 0), thickness=-1)
    # cv2.circle(image, bottom_right, 10, (0, 255, 0), thickness=-1)

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



if __name__ == "__main__":
    # 150889472233113100
    
    image = cv2.imread("data/yolov8_medium-500-pose-simplified-v13/images/train/155321858734451400.jpg")
    lines, bounding_boxes = get_pose_lane_lines("data/yolov8_medium-500-pose-simplified-v13/labels/train/155321858734451400.txt")
    # utils.simplify_lines(lines, tolerance=0.008)


    palette = generate_palette(category_dict)

    draw_lane_lines(image, lines, palette)
    line_id = 1
    print(lines[line_id].points.shape[0])
    offset = 0
    # draw_curve(image, lines[line_id].points[0:2], radius=2, thickness=1)
    # draw_curve(image, lines[line_id].points[1:3], color=(0, 255, 255), radius=2, thickness=1)

    # draw_curve(image, lines[line_id].points, radius=2, thickness=1)
    
    # draw_bounding_boxes(image, bounding_boxes, palette)

    # draw_bounding_box(image, bounding_boxes[0])



    zoom_region = image[700:850, 300:800]

    # cv2.imshow('Image', cv2.resize(zoom_region, None, fx=3, fy=3))
    cv2.imshow('Image', cv2.resize(image, None, fx=0.5, fy=0.5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # idx = 2
    # print(len(lines))
    # print(lines[idx]['nxy'].shape)
    # simplify_lines(lines)
    # print(lines[idx]['nxy'].shape)

    # line = LineString([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    # simplified_line = line.simplify(1.0)

    # # Print the original and simplified lines
    # print(f"Original line: {list(line.coords)}")
    # print(f"Simplified line: {list(simplified_line.coords)}")