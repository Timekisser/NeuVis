import torch
import numpy as np
import ocnn
from ocnn.octree import Octree
from ocnn.octree import Points
from ocnn.nn import OctreeConv, OctreeDeconv
from typing import Dict, List
import torch.nn as nn


class OctreeConvGnRelu(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
        super().__init__()
        self.conv = OctreeConv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.gn = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels) #, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.gn(out)
        out = self.relu(out)
        return out
class OctreeDeconvGnRelu(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
        super().__init__()
        self.deconv = OctreeDeconv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.gn = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels) #, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.deconv(data, octree, depth)
        out = self.gn(out)
        out = self.relu(out)
        return out

# Positional encoding (from NeRF)
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


def get_embedder(multires, input_dims=3):
    if input_dims == -1:
        return torch.nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class UNet(torch.nn.Module):
    def __init__(self, in_channels: int, interp: str = 'linear', nempty: bool = True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.nempty = nempty
        self.config_network()
        self.encoder_stages = len(self.encoder_blocks)
        self.decoder_stages = len(self.decoder_blocks)

    # encoder
        self.conv1 = OctreeConvGnRelu(in_channels, self.encoder_channel[0], nempty=nempty)
        self.downsample = torch.nn.ModuleList([OctreeConvGnRelu(
            self.encoder_channel[i], self.encoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.encoder_stages)])
        self.encoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            self.encoder_channel[i+1], self.encoder_channel[i + 1],
            self.encoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.encoder_stages)])

    # decoder
        channel = [self.decoder_channel[i+1] + self.encoder_channel[-i-2]
                for i in range(self.decoder_stages)]
        self.upsample = torch.nn.ModuleList([OctreeDeconvGnRelu(
            self.decoder_channel[i], self.decoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.decoder_stages)])
        self.decoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            channel[i], self.decoder_channel[i+1],
            self.decoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.decoder_stages)])
        self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)

    def config_network(self):
        self.encoder_channel = [32, 32, 64, 128, 256]
        self.decoder_channel = [256, 256, 128, 96, 96]
        self.encoder_blocks = [2, 3, 4, 6]
        self.decoder_blocks = [2, 2, 2, 2]
        self.bottleneck = 1
        self.resblk = ocnn.modules.OctreeResBlock2

    def unet_encoder(self, data: torch.Tensor, octree: Octree, depth: int):

        convd = dict()
        convd[depth] = self.conv1(data, octree, depth)
        for i in range(self.encoder_stages):
            d = depth - i
            conv = self.downsample[i](convd[d], octree, d)
            convd[d-1] = self.encoder[i](conv, octree, d-1)
        return convd

    def unet_decoder(self, convd: Dict[int, torch.Tensor], octree: Octree, depth: int):
        deconv = convd[depth]
        for i in range(self.decoder_stages):
            d = depth + i
            deconv = self.upsample[i](deconv, octree, d)
            deconv = torch.cat([convd[d+1], deconv], dim=1)  # skip connections
            deconv = self.decoder[i](deconv, octree, d+1)
        return deconv

    def forward(self, data: torch.Tensor, octree: Octree, depth: int, query_pts: torch.Tensor):
        convd = self.unet_encoder(data, octree, depth)
        deconv = self.unet_decoder(convd, octree, depth - self.encoder_stages)

        interp_depth = depth - self.encoder_stages + self.decoder_stages
        feature = self.octree_interp(deconv, octree, interp_depth, query_pts)
        return feature


class VisNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(VisNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp = torch.nn.Sequential(
        ocnn.modules.Conv1x1BnRelu(self.in_channels, 128),
        #ocnn.modules.Conv1x1BnRelu(128, 128),
        ocnn.modules.Conv1x1BnRelu(128, 64),
        #ocnn.modules.Conv1x1BnRelu(64, 64),
        ocnn.modules.Conv1x1(64, self.out_channels, use_bias=True)
        )
    def forward(self, feature):
        output = self.mlp(feature)
        return output
