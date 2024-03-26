import json
import os
import shutil


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


def get_lane_line(json_file, line_idx) -> dict:
    label = category[int(json_file["lane_lines"][line_idx]["category"])]
    u = json_file["lane_lines"][line_idx]["uv"][0]
    v = json_file["lane_lines"][line_idx]["uv"][1]
    return {"label": label,
            "u": u,
            "v": v,
            "points_count": len(u)}


def get_lane_lines(path: str) -> list:
    file = open(path)
    json_file = json.load(file)
    lines_count = len(json_file['lane_lines'])
    
    lines = []
    for line_id in range(lines_count):
        lines.append(get_lane_line(json_file, line_id))
    
    return lines


# Perhaps the coordinates should be normalized. DONE

def convert_label_file_to_yolo(label_file_path, output_path):
    lines = get_lane_lines(label_file_path)
    
    output_string = ""
    for line in lines:
        output_string += str(line['label']) + " "
        for point_idx in range(line['points_count']):
            
            u = line["u"][point_idx] / 1920.0
            v = line["v"][point_idx] / 1280.0
            
            output_string += str(u) + " " + str(v) + " " # may be an error (swap u and v). Also the last space may cause an error.
        output_string += "\n"
    
    with open(output_path, "w") as out:
        out.write(output_string)


def convert_segment_to_yolo(segment_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for full_filename in os.listdir(segment_path):
        filename = full_filename.split(".", 1)[0]
        
        original_file_path = os.path.join(segment_path, full_filename)
        yolo_file_path = os.path.join(output_path, filename + ".txt")
        
        if os.path.isfile(original_file_path):
            convert_label_file_to_yolo(original_file_path, yolo_file_path)


def convert_labels_to_yolo(labels_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for item_name in os.listdir(labels_path):
        full_path = os.path.join(labels_path, item_name)
        output_segment_path = os.path.join(output_path, item_name)
        
        if os.path.isdir(full_path):
            convert_segment_to_yolo(full_path, output_segment_path)


def convert_dataset_to_yolo(dataset_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    if not os.path.exists(output_path + "/labels/train/"):
        os.makedirs(output_path + "/labels/train/")
    
    if not os.path.exists(output_path + "/images/train/"):
        os.makedirs(output_path + "/images/train/")
    
    target_path = os.path.join(output_path, 'labels', 'train')
    labels_path = dataset_path + "/labels/"
    for item_name in os.listdir(labels_path):
        source_path = os.path.join(labels_path, item_name)
        
        
        if os.path.isdir(source_path):
            convert_segment_to_yolo(source_path, target_path)
    
    
    target_path = os.path.join(output_path, 'images', 'train')
    images_path = dataset_path + "/images/"
    for item_name in os.listdir(images_path):
        segment_path = os.path.join(images_path, item_name)
        
        if os.path.isdir(segment_path):
            for image_name in os.listdir(segment_path):
                source_path = os.path.join(segment_path, image_name)
                if os.path.isfile(source_path):
                    shutil.copy(source_path, target_path)
                

if __name__ == "__main__":
    convert_dataset_to_yolo("data/openlane", "data/yolov8")