import json
import os
import shutil
from timeit import default_timer as timer
import src.utils as utils
import sys
import numpy as np


category = {
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


def float_seconds_to_time_str(seconds, decimal_places_to_round_to):
    if seconds < 60.0:
        time = f"{round(seconds, decimal_places_to_round_to)} seconds"
    elif seconds / 60.0 < 60.0:
        time = f"{round(seconds / 60.0, decimal_places_to_round_to)} minutes"
    else:
        time = f"{round((seconds / 60.0) / 60.0, decimal_places_to_round_to)} hours"
    return time


min_points_count = sys.maxsize
max_points_count = 0
mean_points_count = 0
total_lines_count = 0


min_points_count_simplified = sys.maxsize
max_points_count_simplified = 0
mean_points_count_simplified = 0
total_lines_count_simplified = 0

shape_more_then_30 = 0
shape_less_then_30 = 0

max_points_count_simplified_old = 0
max_path = ""

def get_lane_line(json_file, line_idx, shape=(1.0, 1.0)) -> dict:
    label = category[int(json_file["lane_lines"][line_idx]["category"])]
    u = np.array(json_file["lane_lines"][line_idx]["uv"][0])
    v = np.array(json_file["lane_lines"][line_idx]["uv"][1])

    u /= shape[0]
    v /= shape[1]

    if len(u) != len(v):
        return None

    points = np.column_stack((u, v))

    global min_points_count
    global max_points_count
    global mean_points_count
    global total_lines_count

    
    total_lines_count += 1
    min_points_count = min(min_points_count, len(u))
    max_points_count = max(max_points_count, len(u))
    mean_points_count += len(u)
    

    lane_line = utils.LaneLine(points, label)
    utils.simplify_line(lane_line)

    global min_points_count_simplified
    global max_points_count_simplified
    global mean_points_count_simplified
    global total_lines_count_simplified
    global max_points_count_simplified_old
    global shape_more_then_30
    global shape_less_then_30

    max_points_count_simplified_old = max_points_count_simplified

    total_lines_count_simplified += 1
    min_points_count_simplified = min(min_points_count_simplified, lane_line.points.shape[0])

    # if lane_line.points.shape[0] > max_points_count:
    #     max_points_count_simplified = lane_line.points.shape[0]

    max_points_count_simplified = max(max_points_count_simplified, lane_line.points.shape[0])
    mean_points_count_simplified += lane_line.points.shape[0]

    if (lane_line.points.shape[0] > 30):
        shape_more_then_30 += 1
    
    if (lane_line.points.shape[0] <= 30):
        shape_less_then_30 += 1

    return lane_line


def get_bounding_box(line):
    bounding_box = dict()

    bounding_box["label"] = line["label"]

    bounding_box["u"] = (min(line["u"]) + max(line["u"])) / 2.0
    bounding_box["v"] = (min(line["v"]) + max(line["v"])) / 2.0

    bounding_box["width"] = max(line["u"]) - min(line["u"])
    bounding_box["height"] = max(line["v"]) - min(line["v"])

    return bounding_box


def get_normalized_bounding_box(line: utils.LaneLine, shape=[1920.0, 1280.0]) -> dict:
    bounding_box = dict()

    bounding_box["label"] = line.label

    u = line.points[:, 0]
    v = line.points[:, 1]

    u_min = min(u) / shape[0]
    u_max = max(u) / shape[0]

    v_min = min(v) / shape[1]
    v_max = max(v) / shape[1]

    bounding_box["u"] = (u_min + u_max) / 2.0
    bounding_box["v"] = (v_min + v_max) / 2.0

    bounding_box["width"] = u_max - u_min
    bounding_box["height"] = v_max - v_min

    return bounding_box

max_points_count_simplified_test = 0

def get_lane_lines(path: str, verbose=1, shape=(1.0, 1.0)) -> list:
    file = open(path)
    json_file = json.load(file)
    lines_count = len(json_file['lane_lines'])
    
    lines = []
    for line_id in range(lines_count):
        line = get_lane_line(json_file, line_id, shape=shape)

        if max_points_count_simplified_old != max_points_count_simplified:
            global max_path
            global max_points_count_simplified_test
            max_path = path
            max_points_count_simplified_test = max_points_count_simplified

        if line == None:
            if verbose == 1:
                print(f"\033[91mFound corrupt line, id: {0}\033[0m")
            continue

        lines.append(line)
    
    return lines



def convert_label_file_to_yolo(label_file_path, output_path):
    lines = get_lane_lines(label_file_path, shape=(1920.0, 1280.0))
    
    output_string = ""
    for line in lines:
        bounding_box = get_normalized_bounding_box(line)

        output_string += str(line.label) + " " # May be an error
        output_string += str(bounding_box['u']) + " " + str(bounding_box['v']) + " " # May be an error
        output_string += str(bounding_box["width"]) + " " + str(bounding_box["height"]) + " " # May be an error

        # line.points[:, 0] /= 1920.0
        # line.points[:, 1] /= 1280.0

        u = line.points[:, 0]
        v = line.points[:, 1]

        for point_idx in range(line.points.shape[0]): # may be an error
            # u = line["u"][point_idx] / 1920.0
            # v = line["v"][point_idx] / 1280.0

            output_string += str(u[point_idx]) + " " + str(v[point_idx]) + " "
        output_string += "\n"
    
    with open(output_path, "w") as out:
        out.write(output_string)


def convert_segment_to_yolo(segment_path, output_path, max_items=-1, avalible_file_names=None) -> list:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    converted_file_names = []
    
    idx = 0
    for full_filename in os.listdir(segment_path):
        filename = full_filename.split(".", 1)[0]
        
        if avalible_file_names is not None:
            if filename not in avalible_file_names:
                continue
        
        original_file_path = os.path.join(segment_path, full_filename)
        yolo_file_path = os.path.join(output_path, filename + ".txt")
        
        if os.path.isfile(original_file_path):
            if max_items >= 0:
                if idx >= max_items:
                    break
            convert_label_file_to_yolo(original_file_path, yolo_file_path)
            converted_file_names.append(filename)
            idx += 1
    return converted_file_names


def convert_labels_to_yolo(labels_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for item_name in os.listdir(labels_path):
        full_path = os.path.join(labels_path, item_name)
        output_segment_path = os.path.join(output_path, item_name)
        
        if os.path.isdir(full_path):
            convert_segment_to_yolo(full_path, output_segment_path)


def copy_images(segment_path, images_target_path, max_items=-1):
    if os.path.isdir(segment_path):
        idx = 0
        for image_name in os.listdir(segment_path):
            source_path = os.path.join(segment_path, image_name)
            if os.path.isfile(source_path):
                if max_items >= 0:
                    if idx >= max_items:
                        break
                shutil.copy(source_path, images_target_path)
                idx += 1


def get_file_names(path: str, remove_extension=False) -> list:
    files = []
    if os.path.isdir(path):
        for file_name in os.listdir(path):
            
            
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                if remove_extension:
                    file_name = file_name.split('.', 1)[0]
                files.append(file_name)
    return files


def copy_files(source_path, target_path, max_items=-1, avalible_file_names=None):
    
    copied_file_names = []
    
    idx = 0
    for file_name in os.listdir(source_path):
        
        if avalible_file_names is not None:
            no_ext_file_name = file_name.split('.', 1)[0]
            if no_ext_file_name not in avalible_file_names:
                continue
        
        file_path = os.path.join(source_path, file_name)
        if os.path.isfile(file_path):
            if max_items >= 0:
                if idx >= max_items:
                    break
            copied_file_names.append(file_path) 
            shutil.copy(file_path, target_path)
            idx += 1
    return copied_file_names


def delete_files(paths: str):
    for path in paths:
        os.remove(path)


def move_files_old(source_path, target_path, max_items=-1, avalible_file_names=None):
    paths = copy_files(source_path, target_path, max_items=max_items, avalible_file_names=avalible_file_names)
    delete_files(paths)


def move_files(source_path, target_path, max_items=-1):
    if os.path.isdir(target_path):
        idx = 0
        for image_name in os.listdir(source_path):
            full_source_path = os.path.join(source_path, image_name)
            if os.path.isfile(full_source_path):
                if max_items >= 0:
                    if idx >= max_items:
                        break
                shutil.move(full_source_path, target_path)
                print(f"moved: {idx}/{max_items}    path: {full_source_path}")
                idx += 1


def count_files(path: str) -> int:
    files_count = 0
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
            files_count += 1
        elif os.path.isdir(filepath):
            files_count += count_files(filepath)
    return files_count



def convert_dataset_to_yolo(dataset_path, output_path, max_items_count=-1, validation_split=0):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    if not os.path.exists(output_path + "/labels/train/"):
        os.makedirs(output_path + "/labels/train/")
        
    if not os.path.exists(output_path + "/labels/valid/"):
        os.makedirs(output_path + "/labels/valid/")
    
    if not os.path.exists(output_path + "/images/train/"):
        os.makedirs(output_path + "/images/train/")
    
    if not os.path.exists(output_path + "/images/valid/"):
        os.makedirs(output_path + "/images/valid/")
    
    labels_path = dataset_path + "/labels/"
    images_path = dataset_path + "/images/"
    
    image_segments = os.listdir(images_path)
    label_segments = os.listdir(labels_path)
    available_segment_names = set(image_segments) & set(label_segments)
    
    segments_count = len(available_segment_names)
    converted_segments_count = 0
    converted_items_count = 0
    left_items_count = max_items_count
    
    labels_target_path = os.path.join(output_path, 'labels', 'train')
    images_target_path = os.path.join(output_path, 'images', 'train')
    
    valid_labels_target_path = os.path.join(output_path, 'labels', 'valid')
    valid_images_target_path = os.path.join(output_path, 'images', 'valid')
    
    train_split = 1.0 - validation_split

    start = timer()
    for idx, item_name in enumerate(available_segment_names):
        if max_items_count >= 0:
            if left_items_count <= 0:
                break
        
        label_segment_path = os.path.join(labels_path, item_name)
        image_segment_path = os.path.join(images_path, item_name)
        
        if os.path.isdir(label_segment_path) and os.path.isdir(image_segment_path):
            label_names = get_file_names(label_segment_path, True)
            image_names = get_file_names(image_segment_path, True)
            
            available_file_names = set(label_names) & set(image_names)
            
            if len(available_file_names) > 0:
                converted_file_names = convert_segment_to_yolo(label_segment_path, labels_target_path, 
                                                               avalible_file_names=available_file_names, max_items=left_items_count)
                copy_files(image_segment_path, images_target_path, avalible_file_names=converted_file_names)
                converted_segments_count += 1
                converted_items_count += len(converted_file_names)
                
                left_items_count = max_items_count - converted_items_count
                
                end = timer()
                
                elapsed_time = end - start
                eta = (elapsed_time / converted_items_count) * left_items_count
                
                print(f"segments: {converted_segments_count}/{segments_count}   items: {converted_items_count}/{max_items_count}    "
                      f"eta: {float_seconds_to_time_str(eta, 2)}    "
                      f"elapsed: {float_seconds_to_time_str(elapsed_time, 2)}")
    
    # train_items_count = converted_items_count * train_split
    valid_items_count = int(converted_items_count * validation_split)
    
    
    train_labels_count0 = count_files(labels_target_path)
    valid_labels_count0 = count_files(valid_labels_target_path)
    
    train_images_count0 = count_files(images_target_path)
    valid_images_count0 = count_files(valid_images_target_path)
    
    print("AAAAAAAAAAAAAAA")
    move_files(labels_target_path, valid_labels_target_path, max_items=valid_items_count)
    print("BBBBBBBBBBBBBBBBBBB")
    move_files(images_target_path, valid_images_target_path, max_items=valid_items_count)
    
    train_labels_count = count_files(labels_target_path)
    valid_labels_count = count_files(valid_labels_target_path)
    
    train_images_count = count_files(images_target_path)
    valid_images_count = count_files(valid_images_target_path)
    
    total_images_count = train_images_count + valid_images_count
    total_labels_count = train_labels_count + valid_labels_count
    
    print(f"Total images count: {total_images_count}  {(float(total_images_count) / total_images_count)}")
    
    print(f"Train labels count: {train_labels_count}  {(float(train_labels_count) / total_labels_count)}")
    print(f"Train images count: {train_images_count}  {(float(train_images_count) / total_images_count)}")
    
    print(f"Valid labels count: {valid_labels_count}  {(float(valid_labels_count) / total_labels_count)}")
    print(f"Valid images count: {valid_images_count}  {(float(valid_images_count) / total_images_count)}")



if __name__ == "__main__":
    

    # max_points_count = 0
    # # print(count_files("/home/spectre/ProgramFiles/Freedom/LearningProjects/OpenLane/data/openlane/labels"))
    convert_dataset_to_yolo("data/openlane", 
                            "data/yolov8_medium-500-pose-simplified-v14",
                            validation_split=0.2, max_items_count=500)
    print(f"Total lines (full, simpl): ({total_lines_count}, {total_lines_count_simplified})")
    print(f"Min points (full, simpl): ({min_points_count}, {min_points_count_simplified})")
    print(f"Max points (full, simpl): ({max_points_count}, {max_points_count_simplified})")
    print(f"Mean points (full, simpl): ({mean_points_count / float(total_lines_count)}, {mean_points_count_simplified / float(total_lines_count_simplified)})")

    print(f"shape_more_then_30: {shape_more_then_30 / float(total_lines_count)}")
    print(f"shape_less_then_30: {shape_less_then_30 / float(total_lines_count)}")


    print(f"Max path: {max_path}")
    print(f"Max test: {max_points_count_simplified_test}")

    # total_points_count
    