import numpy as np
import open3d as o3d
import os
from tqdm import tqdm

def hpr(vertices, view_point_array):
    # source_data = np.load('data.npz')['vertices']  #10000x3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

    # Define parameters used for hidden_point_removal.
    radius = diameter * 100
    # Get all points that are visible from given view point.
    view_points = [view_point_array[i,:] for i in range(view_point_array.shape[0])]
    labels = []
    for camera in view_points:
        _, pt_map = pcd.hidden_point_removal(camera, radius)
        labels_view = np.zeros(vertices.shape[0])
        labels_view[pt_map] = 1
        labels.append(labels_view)
    labels = np.stack(labels, axis=1)
    return labels
    # pcd = pcd.select_by_index(pt_map)
    # o3d.visualization.draw([pcd], point_size=5)

base_dir = 'data/ShapeNet_NV_simplified'

data = []


for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.npz'):
            data.append(os.path.join(root, file))

ACC = 0
FLAG = 0

pbar_batch = tqdm(total=len(data), ncols=80, leave=False, unit='view')


for each_data in data:
    each_npz = np.load(each_data)
    vertices = each_npz['vertices']
    view_points = each_npz['viewpoints']
    labels_gt = each_npz['visibilities']
    pred = hpr(vertices, view_points)
    accuracy = np.mean(labels_gt == pred) * 100
    pbar_batch.set_postfix({'Accuracy': f'{accuracy:.3f}%'})
    ACC += accuracy
    FLAG += 1
    pbar_batch.update(1)

print(f'预测标签的正确率为: {ACC/FLAG:.3f}%')
 