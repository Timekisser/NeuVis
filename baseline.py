import numpy as np
import open3d as o3d

def hpr(vertices, view_point):
    # source_data = np.load('data.npz')['vertices']  #10000x3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

    # Define parameters used for hidden_point_removal.
    camera = [1, 1, diameter]
    radius = diameter * 100

    # Get all points that are visible from given view point.
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    pcd = pcd.select_by_index(pt_map)
    # o3d.visualization.draw([pcd], point_size=5)
