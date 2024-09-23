# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from thsolver import Dataset
from ocnn.octree import Points
from ocnn.dataset import CollateBatch

from .utils import ReadPly, Transform

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return torch.nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class ShapeNetTransform(Transform):

  def __init__(self, flags):
      super().__init__(flags)
      self.embedder = get_embedder(8)[0]

  def preprocess(self, sample: dict, idx: int):
    xyz = torch.from_numpy(sample['points']).float()
    normal = torch.from_numpy(sample['normals']).float()
    # here we use position encoding on normals(actually viewdir)
    labels = torch.from_numpy(sample['labels']).float()
    viewdirs = self.embedder(normal)
    points = Points(xyz, normal, labels=labels.unsqueeze(1), viewdirs=viewdirs)
    # !NOTE: Normalize the points into one unit sphere in [-0.8, 0.8]
    bbmin, bbmax = points.bbox()
    points.normalize(bbmin, bbmax, scale=0.8)

    return {'points': points}


def get_seg_shapenet_dataset(flags):
  transform = ShapeNetTransform(flags)
  read_ply = ReadPly(has_normal=True, has_label=True)
  collate_batch = CollateBatch(merge_points=True)

  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_ply, take=flags.take)
  return dataset, collate_batch
