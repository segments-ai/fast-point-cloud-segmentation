import subprocess
import numpy as np
import os
import sys

def run_model(dataset_path, output_path):
  dataset_path = os.path.join(os.path.abspath(sys.path[0]), dataset_path)
  output_path = os.path.join(os.path.abspath(sys.path[0]), output_path)
  model_path = os.path.join(os.path.abspath(sys.path[0]), "SSGV3-53")
  script_path = os.path.join(os.path.abspath(sys.path[0]), "SqueezeSegV3/src/tasks/semantic/demo.py")
  wd_path =os.path.join(os.path.abspath(sys.path[0]), "SqueezeSegV3/src/tasks/semantic")
  
  subprocess.call(["python", script_path, "-m", model_path, "-l", output_path, "-d", dataset_path], cwd=wd_path)

def get_prediction(path):
  label = np.fromfile(path, dtype=np.uint32)
  label = label.reshape((-1))
  return label_kitti_to_segments(label)


def label_kitti_to_segments(label):
  instance_ids = label >> 16      # shift 16 bits to the right to get the upper half for instances
  category_ids = label & 0xFFFF   # get lower half for semantics

  unique_cats, counts = np.unique(category_ids, return_counts=True)
  print("Category counts:")
  print(dict(zip(unique_cats, counts)))
  print("Number of points:", len(category_ids))

  annotations = [{"id": i+1, "category_id": int(category_id)} for i, category_id in enumerate(unique_cats)]
  id_dict = {category_id: i+1 for i, category_id in enumerate(unique_cats)}
  point_annotations = [id_dict[category_id] for category_id in category_ids]

  return annotations, point_annotations
