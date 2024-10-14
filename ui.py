import pygame
import sys
import cv2
import math
import numpy as np
from numba import cuda
@cuda.jit
def gpu_render(vertices, img, x_axis, y_axis, z_axis, n_v, H, W, Z, r):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx < n_v:
        vertex = vertices[idx]
        x = vertex[0] * x_axis[0] + vertex[1] * x_axis[1] + vertex[2] * x_axis[2]
        y = vertex[0] * y_axis[0] + vertex[1] * y_axis[1] + vertex[2] * y_axis[2]
        z = vertex[0] * z_axis[0] + vertex[1] * z_axis[1] + vertex[2] * z_axis[2]
        U = int(Z / z * y + H / 2)
        V = int(Z / z * x + W / 2)
        if U >= 0 and U < H and V >= 0 and V < W:
            img[H - U - 1 - math.floor(r / 2):H - U - 1 + math.ceil(r / 2), V - math.floor(r / 2):V + math.ceil(r / 2)] = 255
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
        if t > 1e-3 and 1.0 - t > 1e-3 and u >= 0.0 and v >= 0.0 and (1 - u - v) >= 0.0:
            recorder[v_idx] = True
# Compute the visible points
# When testing the model, write a new function whose input is all vertices np.ndarray[n, 3](dtype=np.float32), and output is visible vertices np.ndarray[n', 3](dtype=np.float32)
def model(vertices, faces, view_point):
    C_device = cuda.to_device(view_point)
    V_device = cuda.to_device(vertices)
    F_device = cuda.to_device(faces)
    recorder_device = cuda.to_device(np.zeros((vertices.shape[0], 1), dtype=bool))
    gpu_intersect[(vertices.shape[0] * F.shape[0]) // 1024 + 1 if(vertices.shape[0] * F.shape[0]) // 1024 + 1 > 1024 else 1024, 1024](C_device, V_device, F_device, vertices.shape[0], faces.shape[0], recorder_device)
    cuda.synchronize()
    result = recorder_device.copy_to_host()
    return vertices[np.where(result == 0)[0]]
def render(vertices, view_point, H, W, Z, r):
    z_axis = -view_point
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, np.array([0, 1, 0], dtype=np.float64))
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(x_axis, z_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    img_device = cuda.to_device(np.zeros((H, W), dtype=np.uint8))
    vertices_device = cuda.to_device(vertices - view_point)
    x_axis_device = cuda.to_device(x_axis)
    y_axis_device = cuda.to_device(y_axis)
    z_axis_device = cuda.to_device(z_axis)
    gpu_render[vertices.shape[0] // 1024 + 1 if vertices.shape[0] // 1024 + 1 > 1024 else 1024, 1024](vertices_device, img_device, x_axis_device, y_axis_device, z_axis_device, vertices.shape[0], H, W, Z, r)
    cuda.synchronize()
    img = img_device.copy_to_host()
    return img
if __name__ == '__main__':
    vertices = np.loadtxt('./model.pts', dtype=np.float32)
    radius = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    file = open('./model_normalized.obj', 'r', encoding='utf-8')
    lines = file.readlines()
    V = []
    f = []
    for line in lines:
        line = line.split()
        if len(line) == 0:
            continue
        if line[0] == 'v' and not line[1] == 'n':
            V.append([float(line[1]), float(line[2]), float(line[3])])
        if line[0] == 'f':
            f.append([int(line[1].split('/')[0]), int(line[2].split('/')[0]), int(line[3].split('/')[0])])
    V = np.asarray(V, dtype=np.float32)
    f = np.asarray(f, dtype=np.int32)
    F = np.zeros((f.shape[0], 3, 3), dtype=np.float32)
    F[:, 0, :] = V[f[:, 0] - 1]
    F[:, 1, :] = V[f[:, 1] - 1]
    F[:, 2, :] = V[f[:, 2] - 1]
    alpha = 0 # (-pi, pi)
    theta = 0 # (0, pi)
    x = radius * np.sin(alpha) * np.cos(theta)
    y = radius * np.sin(theta)
    z = radius * (-np.cos(alpha)) * np.cos(theta)
    view_point = np.array([x, y, z], dtype=np.float32)
    vis_vertices = model(vertices, F, view_point)
    H = 800
    W = 800
    # Compute axis
    z_axis = -view_point
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(z_axis, np.array([0, 1, 0], dtype=np.float64))
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(x_axis, z_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    # Compute the original Z
    x_value = np.dot(vertices - view_point, x_axis)
    y_value = np.dot(vertices - view_point, y_axis)
    z_value = np.dot(vertices - view_point, z_axis)
    Z = np.min([(np.abs(z_value / y_value)).min() * H / 2 - 1, (np.abs(z_value / x_value)).min() * W / 2 - 1])
    Z *= 0.9
    r = 2
    img_org = render(vertices, view_point, H, W, Z, r)
    img_vis = render(vis_vertices, view_point, H, W, Z, r)
    img = np.concatenate([img_org, img_vis], axis=1)
    pygame.init()
    resolution = width,height = W * 2, H 
    windowSurface = pygame.display.set_mode(resolution)
    pygame.display.set_caption("Neural Visibility")
    font = pygame.font.SysFont(None, int(0.03 * H)) 
    text_1 = font.render('Use the arrow keys to control the view direction', True, (255, 255, 255), (0, 0, 0))
    text_2 = font.render('Use the numeric keypad 2, 8 to control the camera field of view', True, (255, 255, 255), (0, 0, 0))
    text_3 = font.render('Use the numeric keypad 4, 6 to control the point size', True, (255, 255, 255), (0, 0, 0))
    text_4 = font.render('Click the numeric kepad 5 to save the current view image', True, (255, 255, 255), (0, 0, 0))
    img_surface = pygame.surfarray.make_surface(img.transpose((1, 0)))
    while True:
        for event in pygame.event.get():
            # 处理退出事件
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                save_flag = 1
                pygame.key.set_repeat(20, 20)
                if event.key == pygame.K_LEFT:
                    alpha += np.pi / 180
                elif event.key == pygame.K_RIGHT:
                    alpha -= np.pi / 180
                elif event.key == pygame.K_UP:
                    theta += np.pi / 180
                elif event.key == pygame.K_DOWN:
                    theta -= np.pi / 180
                elif event.key == pygame.K_KP8:
                    Z *= 1.01
                elif event.key == pygame.K_KP2:
                    Z *= 0.99
                elif event.key == pygame.K_KP4:
                    pygame.key.set_repeat(200, 200)
                    r -= 1
                    if r < 1:
                        r = 1
                elif event.key == pygame.K_KP6:
                    pygame.key.set_repeat(200, 200)
                    r += 1
                elif event.key == pygame.K_KP5:
                    save_flag = 1
                x = radius * np.sin(alpha) * np.cos(theta)
                y = radius * np.sin(theta)
                z = radius * (-np.cos(alpha)) * np.cos(theta)
                view_point = np.array([x, y, z], dtype=np.float32)
                vis_vertices = model(vertices, F, view_point)
                img_org = render(vertices, view_point, H, W, Z, r)
                img_vis = render(vis_vertices, view_point, H, W, Z, r)
                img = np.concatenate([img_org, img_vis], axis=1)
                if save_flag == 1:
                    cv2.imwrite('./compare.png', img)
                img_surface = pygame.surfarray.make_surface(img.transpose((1, 0)))
        windowSurface.blit(img_surface, (0, 0))
        windowSurface.blit(text_1, (0, 0))
        windowSurface.blit(text_2, (0, 1 * int(0.03 * H)))
        windowSurface.blit(text_3, (0, 2 * int(0.03 * H)))
        windowSurface.blit(text_4, (0, 3 * int(0.03 * H)))
        pygame.display.update()