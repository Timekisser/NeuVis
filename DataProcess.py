import os
import numpy as np
from tqdm import tqdm
from numba import cuda
import trimesh
import math

# Mesh Data Format
'''
|directory
|---[category_code]
|---|---[model_code].obj
'''

# Visibility Data Format
'''
|target_directory
|---[category_code]
|---|---[model_code]
|---|---|---data.npz
|---|---|---|---vertices: [(n, 3)-np.float32] (n <= max_npoint)
|---|---|---|---visibilities [(n, n_vp)-bool] (n <= max_npoint)
|---|---|---|---viewpoints [(n_vp, 3)-np.float32]
'''

# max_npoint : target point number of downsampling
# (Downsampling is not performed for pointclouds with more than this number of points)

# n_vp: number of viewpoints for each pointcloud

directory = '/mnt/sdb/wangjh/ShapeNet_Mesh_simplified'
target_directory = './data/ShapeNet_NV_simplified'
max_npoint = 8192
n_vp = 128

@cuda.jit
def gpu_intersect(V, F, f, Vp, n_v, n_f, n_vp, recorder):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_v * n_f * n_vp:
        v_idx = idx // (n_f * n_vp)
        f_idx = (idx - n_f * n_vp * v_idx) // n_vp
        vp_idx = idx - n_f * n_vp * v_idx - n_vp * f_idx
        o = Vp[vp_idx]
        p = V[v_idx]
        v0 = F[f_idx, 0]
        v1 = F[f_idx, 1]
        v2 = F[f_idx, 2]
        # s = o - v0
        s = cuda.local.array(3, dtype=np.float32)
        s[0] = o[0] - v0[0]
        s[1] = o[1] - v0[1]
        s[2] = o[2] - v0[2]
        # e1 = v1 - v0
        e1 = cuda.local.array(3, dtype=np.float32)
        e1[0] = v1[0] - v0[0]
        e1[1] = v1[1] - v0[1]
        e1[2] = v1[2] - v0[2]
        # e2 = v2 - v0
        e2 = cuda.local.array(3, dtype=np.float32)
        e2[0] = v2[0] - v0[0]
        e2[1] = v2[1] - v0[1]
        e2[2] = v2[2] - v0[2]
        # d = p - o
        d = cuda.local.array(3, dtype=np.float32)
        d[0] = p[0] - o[0]
        d[1] = p[1] - o[1]
        d[2] = p[2] - o[2]
        
        d_len = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        #print(d[0], d[1], d[2])
        d[0] = d[0] / d_len
        d[1] = d[1] / d_len
        d[2] = d[2] / d_len
        #print(d[0], d[1], d[2])
        s1 = cuda.local.array(3, dtype=np.float32)
        s2 = cuda.local.array(3, dtype=np.float32)
        # s1 = d x e2
        # s2 = S x e1
        s1[0] = d[1] * e2[2] - d[2] * e2[1]
        s1[1] = d[2] * e2[0] - d[0] * e2[2]
        s1[2] = d[0] * e2[1] - d[1] * e2[0]
        s2[0] = s[1] * e1[2] - s[2] * e1[1]
        s2[1] = s[2] * e1[0] - s[0] * e1[2]
        s2[2] = s[0] * e1[1] - s[1] * e1[0]
        t = (s2[0] * e2[0] + s2[1] * e2[1] + s2[2] * e2[2]) / (s1[0] * e1[0] + s1[1] * e1[1] + s1[2] * e1[2])
        u = (s1[0] * s[0] + s1[1] * s[1] + s1[2] * s[2]) / (s1[0] * e1[0] + s1[1] * e1[1] + s1[2] * e1[2])
        v = (s2[0] * d[0] + s2[1] * d[1] + s2[2] * d[2]) / (s1[0] * e1[0] + s1[1] * e1[1] + s1[2] * e1[2])
        if t >= 1e-5 and d_len - t >= 1e-5 and u >= 0.0 and v >= 0.0 and (1 - u - v) >= 0.0:
            recorder[v_idx, vp_idx] = False

def gt_visibility(V, F, f, Vp):
    V_device = cuda.to_device(V)
    F_device = cuda.to_device(F)
    f_device = cuda.to_device(f)
    Vp_device = cuda.to_device(Vp)
    recorder_device = cuda.to_device(np.ones((V.shape[0], Vp.shape[0]), dtype=bool))
    gpu_intersect[(V.shape[0] * F.shape[0] * Vp.shape[0]) // 1024 + 1 if(V.shape[0] * F.shape[0] * Vp.shape[0]) // 1024 + 1 > 1024 else 1024, 1024](V_device, F_device, f_device, Vp_device, V.shape[0], F.shape[0], Vp.shape[0], recorder_device)
    cuda.synchronize()
    result = recorder_device.copy_to_host()
    return result

def farthest_point_sample(vertices, n_sample=8192):
    indices = np.zeros((n_sample), dtype=np.int32)
    new_vertices = np.zeros((n_sample, 3), dtype=np.float32)
    new_vertices[0] = vertices[0]
    distance = np.linalg.norm(vertices - new_vertices[0], axis=1)
    for i in range(n_sample - 1):
        dist = np.linalg.norm(vertices - new_vertices[i], axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        index = np.argmax(distance)
        indices[i + 1] = index
        new_vertices[i + 1] = vertices[index]
    return new_vertices, indices

catagories = [file for file in os.listdir(os.path.join(directory)) if os.path.isdir(os.path.join(directory, file))]
n_model = 0
models = []
for c in catagories:
    models.append([file[:-4] for file in os.listdir(os.path.join(directory, c)) if os.path.isfile(os.path.join(directory, c, file))])
    n_model += len(models[-1])

pbar = tqdm(total=n_model, ncols=80)

for idx, c in enumerate(catagories):
    for m in models[idx]:
        obj_file = open(os.path.join(directory, c, m + '.obj'), 'r', encoding='utf-8')
        lines = obj_file.readlines()
        V = []
        f = []
        for line in lines:
            line = line.split()
            if len(line) == 0:
                continue
            if line[0] == 'v':
                V.append([float(line[1]), float(line[2]), float(line[3])])
            if line[0] == 'f':
                f.append([int(line[1].split('/')[0]), int(line[2].split('/')[0]), int(line[3].split('/')[0])])
        V = np.asarray(V, dtype=np.float32)
        f = np.asarray(f, dtype=np.int32)
        F = np.zeros((f.shape[0], 3, 3), dtype=np.float32)
        F[:, 0, :] = V[f[:, 0] - 1]
        F[:, 1, :] = V[f[:, 1] - 1]
        F[:, 2, :] = V[f[:, 2] - 1]
        mesh = trimesh.load(os.path.join(directory, c, m + '.obj'))
        new_V = np.asarray(mesh.sample(20000), dtype=np.float32)
        new_V, _ = farthest_point_sample(new_V, max_npoint)
        radius = np.linalg.norm(new_V.max(axis=0) - new_V.min(axis=0))
        alpha = np.random.randn(n_vp) * np.pi * 2 - np.pi
        theta = np.random.randn(n_vp) * np.pi
        alpha = alpha.astype(np.float32)
        theta = theta.astype(np.float32)
        x = radius * np.sin(alpha) * np.cos(theta)
        y = radius * np.sin(theta)
        z = radius * (-np.cos(alpha)) * np.cos(theta)
        Vp = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1, dtype=np.float32)
        new_V_visibility = gt_visibility(new_V, F, f, Vp)
        #print(new_V[new_V_visibility[:, 0]].shape)
        #np.savetxt('./sb.pts', new_V[new_V_visibility[:, 0]])
        if not os.path.exists(os.path.join(target_directory, c, m)):
            os.makedirs(os.path.join(target_directory, c, m))
        np.savez(os.path.join(target_directory, c, m, 'data.npz'), vertices=new_V, visibilities=new_V_visibility, viewpoints=Vp)
        pbar.update(1)