import torch
import numpy as np
import os
import ocnn
import time
from ocnn.octree import Points
from tqdm import tqdm
import torch.nn as nn

from models import UNet, VisNet, get_embedder

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_noise', action="store_true")
parser.add_argument('--test_noise', action='store_true')
parser.add_argument('--alias', type=str, default='naive')
args = parser.parse_args()
train_noise = args.train_noise
test_noise = args.test_noise
alias = args.alias

def points2octree(points, depth=9):
    octree = ocnn.octree.Octree(depth, 2)
    octree.build_octree(points)
    #print(octree.nnum_nempty)
    #print(octree.nnum)
    return octree

def get_input_feature(octree, embedder):
    depth = octree.depth
    #print(depth)
    features = []
    local_points = octree.points[depth].frac() - 0.5
    features.append(local_points)
    scale = 2 ** (1 - depth)   # normalize [0, 2^depth] -> [-1, 1]
    global_points = octree.points[depth] * scale - 1.0
    features.append(global_points)
    # global_points_embedded = embedder(global_points)
    # features.append(global_points_embedded)
    out = torch.cat(features, dim=1)
    return embedder(out)

def process_data(data, sample=32, with_noise=False):
    P = []
    VD = []
    L = []
    for each_data in data:
        each_data_npz = np.load(each_data)
        #vertices = each_data_npz['vertices'] / max(np.abs(each_data_npz['vertices'].min()), np.abs(each_data_npz['vertices'].max())) * 0.8
        index = np.random.permutation(128)
        vertices = torch.from_numpy(each_data_npz['vertices']).float().view(-1, 3)
        if with_noise:
            noise = 0.01*(torch.rand_like(vertices) * 2 - 1)
            vertices = vertices + noise
        view_points = torch.from_numpy(each_data_npz['viewpoints'][index, :]).float().view(-1, 3)
        view_dirs = (vertices.unsqueeze(1) - view_points.unsqueeze(0))
        view_dirs = view_dirs / torch.norm(view_dirs, dim=2).unsqueeze(2)
        VD.append(view_dirs)
        labels = torch.from_numpy(each_data_npz['visibilities'][:, index]).long().unsqueeze(2)
        L.append(labels)
        #print(vertices.min(dim=0), vertices.max(dim=0))
        #print(vertices.shape)
        P.append(Points(vertices))
    VD = torch.concat(VD, dim=0)#.view(-1, 3)
    L = torch.concat(L, dim=0)[:, :sample].reshape(-1, 1)
    P = [p.cuda(non_blocking=True) for p in P]
    O = [points2octree(p) for p in P]
    O = ocnn.octree.merge_octrees(O)
    O.construct_all_neigh()
    P = ocnn.octree.merge_points(P)

    return VD, L, O, P

def model_forward(feature_model, vis_model, oct_embedder, embedder, VD, O, P, sample=32, sub_batch_size=4):
    input_feature = get_input_feature(O, oct_embedder)
    #print(input_feature.shape)
    query_pts = torch.cat([P.points, P.batch_id], dim=1)
    #print(O.depth)
    feature = feature_model(input_feature, O, O.depth, query_pts)
    #print(feature.shape)
    
    output = []
    #print(VD[:, 0 * sub_batch_size:(0 + 1) * sub_batch_size, :].shape)
    for k in range(sample // sub_batch_size):
        sub_feature = feature.unsqueeze(1).repeat(1, sub_batch_size, 1).reshape(-1, 96)
        sub_VD = embedder(VD[:, k * sub_batch_size:(k + 1) * sub_batch_size, :].reshape(-1, 3)).cuda()
        output.append(vis_model(torch.concat([sub_feature, sub_VD], dim=1)).reshape(feature.shape[0], sub_batch_size, -1))
    output = torch.concat(output, dim=1).view(-1, 2)

    return output

base_dir = 'data/ShapeNet_NV/points/02691156'

data = []

embedder, vd_ch = get_embedder(10)
oct_embedder, sg_ch = get_embedder(4, input_dims=6)

feature_model = UNet(sg_ch).cuda()

vis_model = VisNet(96 + vd_ch, 2).cuda()

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.npz'):
            data.append(os.path.join(root, file))

n_data = len(data)

batch_size = 4

n_test = int(((n_data * 0.1) // batch_size) + (n_data % batch_size))
n_train = int(n_data - n_test)
indices = np.random.permutation(n_data)
data = [data[i] for i in indices]

train_data = data[:n_train]
test_data = data[n_train:]

n_epoch = 100

loss_function = nn.CrossEntropyLoss()

optimizer_1 = torch.optim.AdamW(feature_model.parameters(), lr=0.001)
optimizer_2 = torch.optim.AdamW(vis_model.parameters(), lr=0.001)

scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=25, gamma=0.1)
scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=25, gamma=0.1)

test_every_epoch = 1

save_every_epoch = 5

sample = 32

sub_batch_size = 4

for i in range(n_epoch):
    feature_model.train()
    vis_model.train()
    LOSS = 0
    ACCU = 0
    FLAG = 0
    indices = np.random.permutation(n_train)
    train_data = [train_data[k] for k in indices]
    #pbar_epoch.set_description(f'Epoch : {i + 1:03d} / {n_epoch:03d}')
    pbar_batch = tqdm(total=n_train // batch_size, ncols=80, leave=False, unit='batch')
    for j in range(n_train // batch_size):
        #pbar_batch.set_description(f'Batch : {j + 1:03d} / {n_train // batch_size:03d}')
        data_j = train_data[j * batch_size:(j + 1) * batch_size]
        VD, L, O, P = process_data(data_j, sample, train_noise)
        output = model_forward(feature_model, vis_model, oct_embedder, embedder, VD, O, P, sample, sub_batch_size)
        pred = torch.argmax(output, dim=-1, keepdim=True)
        #print(pred.shape, L.shape)
        accu = torch.eq(pred, L.cuda()).float().mean().item()
        #print(accu)
        loss = loss_function(output, L.squeeze(1).cuda())
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(feature_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(vis_model.parameters(), max_norm=1.0)
        optimizer_1.step()
        optimizer_2.step()
        pbar_batch.set_postfix({'Accuracy': f'{accu * 100:.3f}%', 'Loss': f'{loss.item():.5f}'})
        FLAG += 1
        LOSS += loss.item()
        ACCU += accu
        pbar_batch.update(1)
    scheduler_1.step()
    scheduler_2.step()
    pbar_batch.close()
    torch.cuda.empty_cache()
    print(f'EPOCH : {i + 1:03d} / {n_epoch:03d} Accuracy : {ACCU * 100 / FLAG:.3f}% Loss : {LOSS / FLAG:.5f}')
    if (i + 1) % save_every_epoch == 0:
        torch.save(feature_model.state_dict(), f'./checkpoints_3/feature_model_{i + 1:03d}.pth')
        torch.save(vis_model.state_dict(), f'./checkpoints_3/vis_model_{i + 1:03d}.pth')
    if (i + 1) % test_every_epoch == 0:
        feature_model.eval()
        vis_model.eval()
        with torch.no_grad():
            ACCU = 0
            FLAG = 0
            for j in range(n_test):
                data_j = [test_data[j]]
                VD, L, O, P = process_data(data_j, sample=sample, with_noise=test_noise)
                output = model_forward(feature_model, vis_model, oct_embedder, embedder, VD, O, P, sample, sub_batch_size=8)
                pred = torch.argmax(output, dim=-1, keepdim=True)
                accu = torch.eq(pred, L.cuda()).float().mean().item()
                FLAG += 1
                ACCU += accu
            with open(f'log_{alias}.txt', 'a') as fid:
                fid.write(f'TEST : {(i + 1) // test_every_epoch:03d} / {n_epoch // test_every_epoch:03d} Accuracy : {ACCU * 100 / FLAG:.3f}%\n')
                print(f'TEST : {(i + 1) // test_every_epoch:03d} / {n_epoch // test_every_epoch:03d} Accuracy : {ACCU * 100 / FLAG:.3f}%')
        '''     
        output = []
        for k in range(128 // sub_batch_size):
            output = vis_model(vis_input)
        
        print(output.shape, L.shape)
        '''
'''
data = np.load(os.path.join(root, file))
vertices = torch.from_numpy(data['vertices']).float().view(-1, 3)
labels = torch.from_numpy(data['visibilities']).float().view(-1, 1)
view_points = torch.from_numpy(data['viewpoints']).float().view(-1, 3)
view_dirs = (vertices.unsqueeze(1) - view_points.unsqueeze(0)).view(-1, 3)
view_dirs = view_dirs / torch.norm(view_dirs, dim=1).unsqueeze(1)
V.append(vertices)
L.append(labels)
VD.append(view_dirs)
'''
