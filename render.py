import numpy as np
import json
import cv2
from numba import cuda
import os
from tqdm import tqdm
# PointCloud Rasterization on GPU
@cuda.jit
def gpu_render(vertices, img, x_axis, y_axis, z_axis, n_v, H, W, Z):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_v:
        vertex = vertices[idx]
        x = vertex[0] * x_axis[0] + vertex[1] * x_axis[1] + vertex[2] * x_axis[2]
        y = vertex[0] * y_axis[0] + vertex[1] * y_axis[1] + vertex[2] * y_axis[2]
        z = vertex[0] * z_axis[0] + vertex[1] * z_axis[1] + vertex[2] * z_axis[2]
        U = int(Z / z * y + H / 2)
        V = int(Z / z * x + W / 2)
        if U >= 0 and U < H and V >= 0 and V < W:
            img[H - U - 1, V] = 255
# Render PointCloud
def render(all_vertices, vis_vertices, view_point, H=1000, W=1000, pred_vertices=None):
    '''
    all_vertices: np.array[n_1, 3]
    vis_vertices: np.array[n_2, 3]
    pred_vertices: np.array[n_3, 3]
    view_point: np.array[3]
    H: int
    W: int
    '''
    mid_position = np.mean(all_vertices, axis=0)
    # Compute axis
    z_axis = mid_position - view_point
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, np.array([0, 1, 0], dtype=np.float64))
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(x_axis, z_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    # Compute Z
    x_value = np.dot(all_vertices - view_point, x_axis)
    y_value = np.dot(all_vertices - view_point, y_axis)
    z_value = np.dot(all_vertices - view_point, z_axis)
    Z = np.min([(np.abs(z_value / y_value)).min() * H / 2 - 1, (np.abs(z_value / x_value)).min() * W / 2 - 1])
    Z *= 0.9
    # Copy to device
    img_org_device = cuda.to_device(np.zeros((H, W), dtype=np.uint8))
    img_vis_device = cuda.to_device(np.zeros((H, W), dtype=np.uint8))
    all_vertices_device = cuda.to_device(all_vertices - view_point)
    vis_vertices_device = cuda.to_device(vis_vertices - view_point)
    x_axis_device = cuda.to_device(x_axis)
    y_axis_device = cuda.to_device(y_axis)
    z_axis_device = cuda.to_device(z_axis)
    # Render Predict_Vis_Point image
    if pred_vertices is not None:
        pred_vertices_device = cuda.to_device(pred_vertices - view_point)
        img_pred_device = cuda.to_device(np.zeros((H, W), dtype=np.uint8))
        gpu_render[pred_vertices.shape[0] // 1024 + 1 if pred_vertices.shape[0] // 1024 + 1 > 1024 else 1024, 1024](pred_vertices_device, img_pred_device, x_axis_device, y_axis_device, z_axis_device, pred_vertices.shape[0], H, W, Z)
        cuda.synchronize()
    # Render All-Point image
    gpu_render[all_vertices.shape[0] // 1024 + 1 if all_vertices.shape[0] // 1024 + 1 > 1024 else 1024, 1024](all_vertices_device, img_org_device, x_axis_device, y_axis_device, z_axis_device, all_vertices.shape[0], H, W, Z)
    cuda.synchronize()
    # Render GT image
    gpu_render[vis_vertices.shape[0] // 1024 + 1 if vis_vertices.shape[0] // 1024 + 1 > 1024 else 1024, 1024](vis_vertices_device, img_vis_device, x_axis_device, y_axis_device, z_axis_device, vis_vertices.shape[0], H, W, Z)
    cuda.synchronize()
    # Image Concatenate
    img_org = img_org_device.copy_to_host()
    img_vis = img_vis_device.copy_to_host()
    if pred_vertices is not None:
        img_pred = img_pred_device.copy_to_host()
        img = np.concatenate([img_org, img_vis, img_pred], axis=1)
    else:
        img = np.concatenate([img_org, img_vis], axis=1)
    return img
# Run a test on the Dataset
if __name__ == '__main__':
    H = 1000
    W = 1000
    base_dir = "D:\\\\Dataset\\ShapeNet_NeuVis"
    n_model = 0
    for catagory_dir in os.listdir(base_dir):
        for model_dir in os.listdir(os.path.join(base_dir, catagory_dir)):
            n_model += 1
    pbar = tqdm(total=n_model)
    for catagory_dir in os.listdir(base_dir):
        for model_dir in os.listdir(os.path.join(base_dir, catagory_dir)):
            all_vertices = np.loadtxt(os.path.join(base_dir, catagory_dir, model_dir, 'model.pts'), dtype=np.float64)
            vis_vertices = np.loadtxt(os.path.join(base_dir, catagory_dir, model_dir, 'model_visible.pts'), dtype=np.float64)
            with open(os.path.join(base_dir, catagory_dir, model_dir, 'model.json'), 'r+') as f:
                file = json.load(f)
                view_point = np.asarray(file['view_point'], dtype=np.float64)
            img = render(all_vertices, vis_vertices, view_point, H, W)
            cv2.imwrite(os.path.join(base_dir, catagory_dir, model_dir, 'compare.png'), img)
            pbar.update(1)