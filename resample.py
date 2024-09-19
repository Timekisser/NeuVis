import numpy as np
from numba import cuda
import math
import json
import os

# This code creates dataset "ShapeNet_NeuVis"
# The specific structures of the datasets are as follows 

'''
|base_dir
|---ShapeNet_Mesh
|---|---...
|---|---[category_code]
|---|---|---[model_code]
|---|---|---|---models
|---|---|---|---|---model_normalized.obj
|---|---...
|---ShapeNet_Pointcloud
|---|---...
|---|---[category_code]
|---|---|---points
|---|---|---|---[model_code].pts
|---|---...
|---ShapeNet_NeuVis
|---|---...
|---|---[category_code]
|---|---|---[model_code]
|---|---|---|---model.json
|---|---|---|---model.pts
|---|---|---|---model_visible.pts
|---|---...
'''

'''
model.json
{
    "view_point": [3]float,
    "visible_index": [n]int,
}
'''

base_dir = 'D:\\\\Dataset'
mesh_data_dir = 'ShapeNet_Mesh'
cloud_data_dir = 'ShapeNet_Pointcloud'
output_data_dir = 'ShapeNet_NeuVis'

@cuda.jit
def gpu_resample_face(F, n_f, recorder, thresh):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_f:
        p0 = F[idx, 0]
        p1 = F[idx, 1]
        p2 = F[idx, 2]
        a = math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)
        b = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
        c = math.sqrt((p2[0] - p0[0])**2 + (p2[1] - p0[1])**2 + (p2[2] - p0[2])**2)
        p = 0.5 * (a + b + c)
        S = math.sqrt(p * (p - a) * (p - b) * (p - c))
        if S > thresh:
            num = math.floor(math.sqrt(S / thresh))
            recorder[idx] = num

@cuda.jit
def gpu_downsample(V, n_v, recorder, x, y, z, d):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_v:
        P = V[idx]
        recorder[int(math.floor((P[0] - x) / d)), int(math.floor((P[1] - y) / d)), int(math.floor((P[2] - z) / d))] = idx


@cuda.jit
def gpu_resample_line(L, n_l, recorder, thresh):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_l:
        p0 = L[idx, 0]
        p1 = L[idx, 1]
        length = math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)
        if length > thresh:
            num = math.floor(length / thresh)
            recorder[idx] = num

@cuda.jit
def gpu_intersect(c, v, F, n_v, n_f, recorder):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_v * n_f:
        
        v_idx = idx // n_f
        f_idx = idx - n_f * v_idx
        
        o = c
        
        p = v[v_idx]

        v0 = F[f_idx, 0]
        v1 = F[f_idx, 1]
        v2 = F[f_idx, 2]
        
        # s = o - v0
        s = cuda.local.array(3, dtype=np.float64)
        s[0] = o[0] - v0[0]
        s[1] = o[1] - v0[1]
        s[2] = o[2] - v0[2]
        
        # e1 = v1 - v0
        e1 = cuda.local.array(3, dtype=np.float64)
        e1[0] = v1[0] - v0[0]
        e1[1] = v1[1] - v0[1]
        e1[2] = v1[2] - v0[2]
        
        # e2 = v2 - v0
        e2 = cuda.local.array(3, dtype=np.float64)
        e2[0] = v2[0] - v0[0]
        e2[1] = v2[1] - v0[1]
        e2[2] = v2[2] - v0[2]
        
        # d = p - o
        d = cuda.local.array(3, dtype=np.float64)
        d[0] = p[0] - o[0]
        d[1] = p[1] - o[1]
        d[2] = p[2] - o[2]
        
        s1 = cuda.local.array(3, dtype=np.float64)
        s2 = cuda.local.array(3, dtype=np.float64)
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
        
        if t > 1e-3 and 1.0 - t > 1e-3 and u >= 0.0 and v >= 0.0 and (1 - u - v) >= 0.0:
            recorder[v_idx] = True

catagories = [file for file in os.listdir(os.path.join(base_dir, mesh_data_dir)) if os.path.isdir(os.path.join(base_dir, mesh_data_dir, file))]

n_model = 0
models = []
for c in catagories:
    models.append([file for file in os.listdir(os.path.join(base_dir, mesh_data_dir, c)) if os.path.isdir(os.path.join(base_dir, mesh_data_dir, c, file))])
    n_model += len(models[-1])

iteration = 0

for idx, c in enumerate(catagories):
    for m in models[idx]:
        iteration += 1
        
        file = os.path.join(base_dir, mesh_data_dir, c, m, 'models', 'model_normalized.obj')
        
        V = []
        f = []
        l = []
        if not os.path.exists(file):
            continue
        file = open(file, 'r', encoding='utf-8')
        lines = file.readlines()
        for line in lines:
            line = line.split()
            if len(line) == 0:
                continue
            if line[0] == 'v' and not line[1] == 'n':
                V.append([float(line[1]), float(line[2]), float(line[3])])
            if line[0] == 'f':
                f.append([int(line[1].split('/')[0]), int(line[2].split('/')[0]), int(line[3].split('/')[0])])
            if line[0] == 'l':
                l.append([int(line[1]), int(line[2])])
        V = np.asarray(V, dtype=np.float64)
        f = np.asarray(f, dtype=np.int64)
        F = np.zeros((f.shape[0], 3, 3), dtype=np.float64)
        F[:, 0, :] = V[f[:, 0] - 1]
        F[:, 1, :] = V[f[:, 1] - 1]
        F[:, 2, :] = V[f[:, 2] - 1]
        F_device = cuda.to_device(F)
        recorder_device = cuda.to_device(np.zeros((f.shape[0]), dtype=np.int64))
        gpu_resample_face[(f.shape[0] // 1024 + 1) if (f.shape[0] // 1024 + 1) > 1024 else 1024, 1024](F_device, f.shape[0], recorder_device, 0.000016)
        cuda.synchronize()
        recorder = recorder_device.copy_to_host()

        flag1 = V.shape[0]
        flag2 = 0
        flag3 = 0
        
        V1 = []
        for idx, num in enumerate(recorder):
            if num == 0:
                continue
            v0 = F[idx, 0]
            v1 = F[idx, 1]
            v2 = F[idx, 2]
            u = np.sqrt(np.random.random(num)).reshape((-1, 1))
            v = np.random.random(num).reshape((-1, 1))
            V1.append(v0 * (1.0 - u) + v1 * (u * (1.0 - v)) + v2 * (u * v))
        if len(V1) == 1:
            V1 = V1[0]
            V = np.concatenate([V, V1], axis=0)
            flag2 = V1.shape[0]
        elif len(V1) > 1:
            V1 = np.concatenate(V1, axis=0)
            V = np.concatenate([V, V1], axis=0)
            flag2 = V1.shape[0]

        if len(l) > 0:
            l = np.asarray(l, dtype=np.int64)
            L = np.zeros((l.shape[0], 2, 3), dtype=np.float64)
            L[:, 0, :] = V[l[:, 0] - 1]
            L[:, 1, :] = V[l[:, 1] - 1]
            L_device = cuda.to_device(L)
            recorder_device = cuda.to_device(np.zeros((l.shape[0]), dtype=np.int64))
            gpu_resample_line[(l.shape[0] // 1024 + 1) if (l.shape[0] // 1024 + 1) > 1024 else 1024, 1024](L_device, l.shape[0], recorder_device, 0.004)
            cuda.synchronize()
            recorder = recorder_device.copy_to_host()
            V2 = []
            for idx, num in enumerate(recorder):
                if num == 0:
                    continue
                v0 = L[idx, 0]
                v1 = L[idx, 1]
                d = v1 - v0
                x = (np.arange(num + 1) / (num + 1))[1:].reshape((-1, 1))
                V2.append(v0 + x * d)
            if len(V2) == 1:
                V2 = V2[0]
                V = np.concatenate([V, V2], axis=0)
                flag3 = V2.shape[0]
            elif len(V2) > 1:
                V2 = np.concatenate(V2, axis=0)
                V = np.concatenate([V, V2], axis=0)
                flag3 = V2.shape[0]
        
        print('(', iteration, '/', n_model,  ')')
        print(flag1 + flag2 + flag3, '=', flag1, '+', flag2, '+', flag3)
        
        d = 0.01
        x, y, z = V.min(axis=0) - 1e-4
        V_device = cuda.to_device(V)
        recorder_device = cuda.to_device(np.zeros((np.ceil((V.max(axis=0) - V.min(axis=0)) / d) + 1).astype(np.int64)) - 1)
        gpu_downsample[V.shape[0] // 1024 + 1 if V.shape[0] // 1024 + 1 > 1024 else 1024, 1024](V_device, V.shape[0], recorder_device, x, y, z, d)
        cuda.synchronize()
        recorder = recorder_device.copy_to_host()
        recorder = recorder.reshape(-1)
        recorder = recorder[np.where(recorder > -1)[0]]
        new_V = V[recorder.astype(np.int64)]
        V = new_V
        
        if V.shape[0] < 2000:
            print('WARNING: TOO FEW POINTS', V.shape[0])
            continue
        
        radius = np.max(V.max(axis=0) - V.min(axis=0))
        
        alpha = np.random.random(1)[0] * np.pi * 2 - np.pi
        theta = np.random.random(1)[0] * np.pi
        x = radius * np.sin(theta) * np.cos(alpha)
        y = radius * np.sin(theta) * np.sin(alpha)
        z = radius * np.cos(theta)
        view_point = np.array([x, y, z], dtype=np.float64)
        
        C_device = cuda.to_device(view_point)
        V_device = cuda.to_device(V)
        F_device = cuda.to_device(F)
        recorder_device = cuda.to_device(np.zeros((V.shape[0], 1), dtype=bool))
        gpu_intersect[(V.shape[0] * F.shape[0]) // 1024 + 1 if(V.shape[0] * F.shape[0]) // 1024 + 1 > 1024 else 1024, 1024](C_device, V_device, F_device, V.shape[0], F.shape[0], recorder_device)
        cuda.synchronize()
        result = recorder_device.copy_to_host()
        indices = np.where(result == 0)[0]
        
        if not os.path.exists(os.path.join(base_dir, output_data_dir, c, m)):
            os.makedirs(os.path.join(base_dir, output_data_dir, c, m))
        text = {}
        text['view_point'] = view_point.tolist()
        text['visible_index'] = indices.tolist()
        with open(os.path.join(base_dir, output_data_dir, c, m, 'model.json'), 'w') as f:
            json.dump(text, f)
        np.savetxt(os.path.join(base_dir, output_data_dir, c, m, 'model.pts'), V)
        np.savetxt(os.path.join(base_dir, output_data_dir, c, m, 'model_visible.pts'), V[indices])
        print(V.shape[0], V[indices].shape[0])