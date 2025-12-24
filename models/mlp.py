"""
This file contains the MLP models for the Liver HyperDiffusion.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2025-12-24

Disclaimer: This file is taken and modified from many sources.
"""

import torch
from torch import nn
import torch.nn.functional as F


### -------------------------------------------------------- From HyperDiffusion -------------------------------------------------------- ###
### They use the Position Encoding (PE) of NeRF. Doesn't have the `pi` factor in the encoding.
class Embedder:
    def __init__(self, 
                 input_dims,
                 include_input=True, 
                 max_freq=10,
                 num_freqs=10, 
                 log_sampling=True,
                 periodic_fns=[torch.sin, torch.cos]):
        """Position Encoding (PE) class (section 5.1) from NeRF-PyTorch implementation
        https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py#L15

        Apply position encoding to the input coordinates x:
        embedding = PE(x)

        The encoding transforms the input coordinates as follows:
        "x -> [sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]"

        If include_input is True, the raw input is also included in the encoding:
        "x -> [x, sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^(L-1) * pi * x), cos(2^(L-1) * pi * x)]"  
        
        Args:
            input_dims (int): Number of input dimensions to encode
            include_input (bool, optional): Whether to include the raw input in the encoding. Defaults to True.
            max_freq (int, optional): Maximum frequency in log2 space. Defaults to 10.
            num_freqs (int, optional): Number of frequency bands to sample. Defaults to 10.
            log_sampling (bool, optional): Whether to sample frequencies logarithmically. Defaults to True.
            periodic_fns (list, optional): List of periodic functions to use for encoding. Defaults to [torch.sin, torch.cos].
        """
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq = max_freq
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """Create the embedding functions and calculate output dimension
        """
        embed_fns = []
        d = self.input_dims
        out_dim = 0

        ### Add raw input if specified
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        ### Sample frequencies either logarithmically or linearly
        if self.log_sampling:
            ### Logarithmic sampling concentrates more bands at lower frequencies
            ### freq_bands = [2^0, 2^1, 2^2, ..., 2^max_freq]
            ### len = num_freqs (=4 in HyperDiffusion)
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq, steps=self.num_freqs)
        else:
            ### Linear sampling spaces frequencies uniformly
            ### freq_bands = linear spacing from 2^0 to 2^max_freq
            ### len = num_freqs (=4 in HyperDiffusion)
            freq_bands = torch.linspace(2.0**0.0, 2.0**self.max_freq, steps=self.num_freqs)

        ### Create embedding functions for each frequency and periodic function
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                ### p_fn(x * freq) is missing the `pi` factor in this implementation.
                ### Due to the original NeRF paper implementation does not have the `pi` factor too.
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """
        Embed the input coordinates using the position encoding functions.

        Given the input is 3D coordinates, so inputs.shape = (B, 3) with B is the batch size and 3 is the x, y, z coordinates.
        Assume multires=4 --> num_freqs=4 --> 4*Sin + 4*Cos = 8 frequency bands.
        We will apply the 8 frequency bands to the x, y, z coordinates --> 3 * 8 = 24
        If include_input=True, we will also include the raw input --> 24 + 3 = 27
        """
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)



class MLP3D(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_neurons,
        use_leaky_relu=False,
        use_bias=True,
        include_input=True,
        multires=4,
        log_sampling=True,
    ):
        super().__init__()

        ### Position Encoding (PE)
        self.embedder = Embedder(
            input_dims=input_size, ### 3 if not move else 4,
            include_input=include_input,
            max_freq=multires - 1,
            num_freqs=multires,
            log_sampling=log_sampling,
            periodic_fns=[torch.sin, torch.cos],
        )

        self.use_leaky_relu = use_leaky_relu
        in_size = self.embedder.out_dim

        ### Create MLP
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], output_size, bias=use_bias))

    def forward(self, coords):
        x = self.embedder.embed(coords)

        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.leaky_relu(x) if self.use_leaky_relu else F.relu(x)
        x = self.layers[-1](x)

        return x

