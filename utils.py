import subprocess
import numpy as np
import os
import sys


def get_kitti_attributes():
    labels = {
        1: "outlier",
        10: "car",
        11: "bicycle",
        13: "bus",
        15: "motorcycle",
        16: "on-rails",
        18: "truck",
        20: "other-vehicle",
        30: "person",
        31: "bicyclist",
        32: "motorcyclist",
        40: "road",
        44: "parking",
        48: "sidewalk",
        49: "other-ground",
        50: "building",
        51: "fence",
        52: "other-structure",
        60: "lane-marking",
        70: "vegetation",
        71: "trunk",
        72: "terrain",
        80: "pole",
        81: "traffic-sign",
        99: "other-object",
        252: "moving-car",
        253: "moving-bicyclist",
        254: "moving-person",
        255: "moving-motorcyclist",
        256: "moving-on-rails",
        257: "moving-bus",
        258: "moving-truck",
        259: "moving-other-vehicle",
    }

    color_map = {
        0: [0, 0, 0],
        1: [0, 0, 255],
        10: [245, 150, 100],
        11: [245, 230, 100],
        13: [250, 80, 100],
        15: [150, 60, 30],
        16: [255, 0, 0],
        18: [180, 30, 80],
        20: [255, 0, 0],
        30: [30, 30, 255],
        31: [200, 40, 255],
        32: [90, 30, 150],
        40: [255, 0, 255],
        44: [255, 150, 255],
        48: [75, 0, 75],
        49: [75, 0, 175],
        50: [0, 200, 255],
        51: [50, 120, 255],
        52: [0, 150, 255],
        60: [170, 255, 150],
        70: [0, 175, 0],
        71: [0, 60, 135],
        72: [80, 240, 150],
        80: [150, 240, 255],
        81: [0, 0, 255],
        99: [255, 255, 50],
        252: [245, 150, 100],
        256: [255, 0, 0],
        253: [200, 40, 255],
        254: [30, 30, 255],
        255: [90, 30, 150],
        257: [250, 80, 100],
        258: [180, 30, 80],
        259: [255, 0, 0],
    }

    categories = [
        {"id": key, "name": value, "color": color_map[key]}
        for (key, value) in labels.items()
    ]

    task_attributes = {
        "format_version": "0.1",
        "categories": categories,
    }

    return task_attributes


def run_model(dataset_path, output_path):
    dataset_path = os.path.join(os.path.abspath(sys.path[0]), dataset_path)
    output_path = os.path.join(os.path.abspath(sys.path[0]), output_path)
    model_path = os.path.join(os.path.abspath(sys.path[0]), "SSGV3-53")
    script_path = os.path.join(
        os.path.abspath(sys.path[0]), "SqueezeSegV3/src/tasks/semantic/demo.py"
    )
    wd_path = os.path.join(
        os.path.abspath(sys.path[0]), "SqueezeSegV3/src/tasks/semantic"
    )

    subprocess.call(
        [
            "python",
            script_path,
            "-m",
            model_path,
            "-l",
            output_path,
            "-d",
            dataset_path,
        ],
        cwd=wd_path,
    )


def get_prediction(path):
    label = np.fromfile(path, dtype=np.uint32)
    label = label.reshape((-1))
    return label_kitti_to_segments(label)


def label_kitti_to_segments(label):
    instance_ids = (
        label >> 16
    )  # shift 16 bits to the right to get the upper half for instances
    category_ids = label & 0xFFFF  # get lower half for semantics

    unique_cats = np.unique(category_ids)
    unique_instances, indices = np.unique(instance_ids, return_index=True)
    instances_cats = [category_ids[indices[i]] for i in range(len(unique_instances))]

    annotations = []
    instance_id = 1
    cat_map = {0: 0}

    for cat in unique_cats:
        if cat == 0:
            continue

        for i, instances_cat in enumerate(instances_cats):
            if instances_cat == cat and unique_instances[i] > 0:
                annotations.append(
                    {"id": int(unique_instances[i]), "category_id": int(instances_cat)}
                )
        while instance_id in unique_instances:
            instance_id += 1

        annotations.append({"id": instance_id, "category_id": int(cat)})
        cat_map[cat] = instance_id
        instance_id += 1

    point_annotations = [
        cat_map[category_ids[i]] if id == 0 else int(id)
        for i, id in enumerate(instance_ids)
    ]

    return annotations, point_annotations
