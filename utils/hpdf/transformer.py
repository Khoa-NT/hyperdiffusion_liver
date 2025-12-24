"""
This file contains the G.pt model and its building blocks (minGPT without masking, etc.).
"""
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F



class SelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Modified from: https://github.com/karpathy/minGPT
    A version of minGPT without masking (we use diffusion instead).
    """

    def __init__(
        self,
        input_parameter_sizes,  ### [[3456, 128, 16384, 128, 16384, 128, 128, 1], [257]]
        output_parameter_sizes, ### [[3456, 128, 16384, 128, 16384, 128, 128, 1]]
        input_parameter_names,  ### ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias', 'layers.2.weight', 'layers.2.bias', 'layers.3.weight', 'layers.3.bias', 'timestep_embedding']
        n_layer=12,         ### n_layer: 12 in train_plane.yaml
        n_head=12,          ### n_head: 16 in train_plane.yaml
        n_embd=768,         ### n_embd: 2880 in train_plane.yaml
        encoder_depth=1,
        decoder_depth=1,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        chunk_size=None,
        split_policy="chunk", ### split_policy: layer_by_layer in train_plane.yaml
    ):
        # parameter_sizes is a list of integers indicating how many parameters are in each layer
        super().__init__()
        # print(f"\nFrom {self.__class__.__name__} __init__")

        # Determine how many parameters are placed into each individual Transformer token:
        ### The split_policy="layer_by_layer" in config file
        ### Convert the nested list (input_parameter_sizes) to a flat list (self.input_splits)
        ### E.g., input_parameter_sizes = [[3456, 128, 16384, 128, 16384, 128, 128, 1], [257]]
        ### self.input_splits = [3456, 128, 16384, 128, 16384, 128, 128, 1, 257]
        self.input_splits = self.build_splits(
            input_parameter_sizes, split_policy, chunk_size
        )
        ### The same as input_parameter_sizes
        ### Convert the nested list (output_parameter_sizes) to a flat list (self.output_splits)
        ### E.g., output_parameter_sizes = [[3456, 128, 16384, 128, 16384, 128, 128, 1]]
        ### self.output_splits = [3456, 128, 16384, 128, 16384, 128, 128, 1]
        self.output_splits = self.build_splits(
            output_parameter_sizes, split_policy, chunk_size
        )
        # print(f"Using following input parameter splits: {self.input_splits}")   ### [3456, 128, 16384, 128, 16384, 128, 128, 1, 257]
        # print(f"Using following output parameter splits: {self.output_splits}") ### [3456, 128, 16384, 128, 16384, 128, 128, 1]
        
        ### The number of items (Weights, biases, etc.) in the input_parameter_names
        block_size = len(self.input_splits)
        # print(input_parameter_names)
        if split_policy == "layer_by_layer":
            assert len(input_parameter_names) == block_size
        else:
            input_parameter_names = ["null"] * block_size

        # input embedding stem
        ### positional embedding for each token in transformer
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        ### Transformer blocks (SelfAttention + MLP)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, attn_pdrop, resid_pdrop) for _ in range(n_layer)]
        )
        self.block_size = block_size

        ### Create the projection layer for each token before feed into the Transformer blocks
        ### Input_weight --MLP--> n_embd
        # Per-token encoder layers:
        self.input_parameter_projections = self.build_encoder(
            n_embd, encoder_depth, self.input_splits
        )
        self.ln_in = nn.LayerNorm(n_embd)

        ### Create the projection layer for each token after feed into the Transformer blocks
        ### n_embd --MLP--> output_parameter_sizes
        ### self.output_splits = [3456, 128, 16384, 128, 16384, 128, 128, 1]
        # Per-token decoder layers:
        self.ln_f = nn.LayerNorm(n_embd)
        self.output_parameter_projections = self.build_decoder(
            n_embd, decoder_depth, self.output_splits
        )

        ### E.g., self.output_splits = [3456, 128, 16384, 128, 16384, 128, 128, 1]
        self.num_output_heads = len(self.output_splits)
        self.apply(self._init_weights)

        # print(f"number of parameters: {sum(p.numel() for p in self.parameters()):,}")

    @staticmethod
    def build_encoder(n_embd, encoder_depth, input_splits):
        # Create a unique MLP encoder for each token
        input_parameter_projections = nn.ModuleList()
        
        ### self.input_splits = [3456, 128, 16384, 128, 16384, 128, 128, 1, 257]
        for param_chunk_size in input_splits:
            in_proj = [nn.Linear(param_chunk_size, n_embd, bias=False)]
            for _ in range(encoder_depth - 1):
                in_proj.append(nn.GELU())
                in_proj.append(nn.Linear(n_embd, n_embd, bias=False))
            in_proj = nn.Sequential(*in_proj)
            input_parameter_projections.append(in_proj)

        return input_parameter_projections

    @staticmethod
    def build_decoder(n_embd, decoder_depth, output_splits):
        # Create a unique MLP decoder for each noised token
        output_parameter_projections = nn.ModuleList()

        ### self.output_splits = [3456, 128, 16384, 128, 16384, 128, 128, 1]
        for output_chunk_size in output_splits:
            out_proj = []

            ### decoder_depth: 1 in train_plane.yaml
            ### It means this for loop will not run
            for _ in range(decoder_depth - 1): 
                out_proj.append(nn.Linear(n_embd, n_embd, bias=False))
                out_proj.append(nn.GELU())

            ### Create MLP for each token (Weights or Biases, etc individually)
            out_proj.append(nn.Linear(n_embd, output_chunk_size, bias=False))
            out_proj = nn.Sequential(*out_proj)
            output_parameter_projections.append(out_proj)

        return output_parameter_projections

    ### Doesn't seem to be used in the code
    def get_block_size(self):
        return self.block_size

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def configure_optimizers(nn_module, lr, wd, betas):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            FrequencyEmbedder,
        )
        for mn, m in nn_module.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif (
                    pn.endswith("pos_emb")
                    or pn.endswith("cfg_loss_embedding")
                    or pn.endswith("hypernet_z_tokens")
                ):
                    # special case the position embedding parameter
                    # in the root GPT module as not decayed
                    no_decay.add(fpn)
        # decay.add('decoder._fsdp_wrapped_module.flat_param')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in nn_module.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": wd,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer

    def encode_parameters(self, parameters):
        """
        Chunk input parameter vector, apply per-chunk encoding, and
        stack projected chunks along the sequence (token) dimension.
        """
        assert parameters.dim() == 2

        ### With self.input_splits = [3456, 128, 16384, 128, 16384, 128, 128, 1, 257]
        ### Split the parameters into 9 chunks
        split_parameters = torch.split(parameters, self.input_splits, dim=1)

        representations = []
        for parameter, in_proj in zip(split_parameters, self.input_parameter_projections):
            ### Apply individual MLP to each chunk (Weights or Biases, etc.)
            representations.append(in_proj(parameter))

        representations = torch.stack(representations, dim=1)  # (b, t, d)
        representations = self.ln_in(representations)
        assert representations.dim() == 3  
        return representations ### [B, token length, n_embd] ### [32, 9, 2880]

    def decode_parameters(self, features):
        """
        Apply a per-chunk decoding (only to the tokens corresponding to the noised/updated parameter vector),
        and concatenate them into a flattened parameter vector.
        """
        assert features.dim() == 3  # (b, t, d) ### [B, token length, n_embd] ### [32, 9, 2880]
        output = []

        ### E.g., self.output_splits = [3456, 128, 16384, 128, 16384, 128, 128, 1]
        ### self.num_output_heads = len(self.output_splits) = 8
        for t in range(self.num_output_heads):
            ### Get the MLP corresponding to the t-th token
            out_proj = self.output_parameter_projections[t]

            ### Apply the MLP to the t-th token
            output.append(out_proj(features[:, t, :]))

        ### Concatenate the output of all the MLPs
        ### output is a list of [B, output_chunk_size]
        output = torch.cat(output, 1)  # (b, c)
        assert output.dim() == 2

        return output

    @staticmethod
    def flatten_parameter_sizes(parameter_sizes):
        assert len(parameter_sizes) > 0
        if not isinstance(parameter_sizes[0], (list, tuple)):  # list is already flat
            return parameter_sizes
        return [p for group in parameter_sizes for p in group]

    @staticmethod
    def build_splits(parameter_sizes, split_policy="chunk", chunk_size=None):
        """
        Determines how to split the input parameter vector into individual tokens.

        'chunk': Basic approach (naively concatenate all inputs together and chunk them
                 indiscriminately, each token can contain values from different layers and inputs)
        'layer_by_layer': each layer's parameters are contained in a SINGLE token,
                          no mixing across layers or inputs
        'chunk_within_layer': each layer's parameters are subdivided into MANY tokens,
                              no mixing across layers or inputs
        'chunk_within_input': each input's elements are subdivided into MANY tokens,
                              no mixing across inputs
        """
        if split_policy == "chunk":
            # Chunk the parameter vector, not caring if one chunk contains parameters
            # from different layers:
            assert chunk_size is not None
            parameter_sizes = GPT.flatten_parameter_sizes(parameter_sizes)
            total_n_params = sum(parameter_sizes)
            num = total_n_params // chunk_size
            splits = [chunk_size] * num
            remainder = total_n_params % chunk_size
            if remainder > 0:
                splits.append(remainder)
        elif split_policy == "layer_by_layer":
            # Each layer's parameters belong to its own chunk:
            parameter_sizes = GPT.flatten_parameter_sizes(parameter_sizes)
            splits = parameter_sizes
        elif split_policy == "chunk_within_layer":
            # Chunk the parameter vector, ensuring that each chunk contains parameters
            # from a single layer only:
            assert chunk_size is not None
            parameter_sizes = GPT.flatten_parameter_sizes(parameter_sizes)
            splits = []
            for param_size in parameter_sizes:
                num = param_size // chunk_size
                splits.extend([chunk_size] * num)
                remainder = param_size % chunk_size
                if remainder > 0:
                    splits.append(remainder)
        elif split_policy == "chunk_within_input":
            splits = []
            for parameter_group in parameter_sizes:
                assert isinstance(parameter_group, (list, tuple))
                splits.extend(GPT.build_splits(parameter_group, "chunk", chunk_size))
            return splits
        else:
            raise NotImplementedError
        return splits

    def forward(self, x):
        # print(f"\nFrom {self.__class__.__name__}")
        # print(f"x.shape: {x.shape}") ### [32, 36994] ### Batch size = 100 and 36994 = 36737 (parameters) + 257 (timestep embedding)

        ### Encode the parameters --MLP--> n_embd
        ### Which means apply MLP to each chunk of `[3456, 128, 16384, 128, 16384, 128, 128, 1, 257]` --> `[n_embd]*9`
        embeddings = self.encode_parameters(x)
        # print(f"embeddings.shape: {embeddings.shape}") ### [32, 9, 2880] ### [B, token length, n_embd] ### token length = 9 = 8 (weights or biases individually) + 1 (timestep embedding)

        ### Batch size, number of Weights and Biases, embedding dimension
        b, t, d = embeddings.size()
        # print(f"b: {b}, t: {t}, d: {d}") ### b: 32, t: 9, d: 2880

        ### Check if the number of tokens is correct
        assert (
            t == self.block_size
        ), f"Expected {self.block_size} tokens on dim=1, but got {t}"

        # forward the GPT model
        ### Get the positional embedding for each token 
        ### from pos_emb (1, n_weight, n_embd)
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        # print(f"position_embeddings.shape: {position_embeddings.shape}") ### [1, 9, 2880]

        ### Then sum the embeddings and the positional embeddings
        x = self.drop(embeddings + position_embeddings)
        # print(f"After `self.drop(embeddings + position_embeddings)`, x.shape: {x.shape}") ### [32, 9, 2880]

        ### Pass the embeddings through the Transformer blocks
        x = self.blocks(x)
        # print(f"After `self.blocks(x)`, x.shape: {x.shape}") ### [32, 9, 2880]

        ### Normalize the embeddings
        x = self.ln_f(x)
        # print(f"After `self.ln_f(x)`, x.shape: {x.shape}") ### [32, 9, 2880]
        
        ### Decode the embeddings --MLP--> output_parameter_sizes
        ### Convert the embeddings back to the original parameter sizes on the parameter-tokens, except for the timestep embedding.
        ### E.g., self.output_splits = [3456, 128, 16384, 128, 16384, 128, 128, 1]
        ### each parameter-token is given into a MLP individually (1 MLP per parameter-token, not shared)
        ### then concatenate all the MLPs' outputs together [3456, 128, 16384, 128, 16384, 128, 128, 1] --> [36737]
        x = self.decode_parameters(x)
        # print(f"After `self.decode_parameters(x)`, x.shape: {x.shape}") ### [32, 36737] ### [B, total_parameters]

        return x


class FrequencyEmbedder(nn.Module):
    def __init__(self, num_frequencies, max_freq_log2):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq_log2, steps=num_frequencies)
        self.register_buffer("frequencies", frequencies)
        # self.device = device

    def forward(self, x):
        # x should be of size (N,) or (N, D)
        N = x.size(0)
        if x.dim() == 1:  # (N,)
            x = x.unsqueeze(1)  # (N, D) where D=1
        # x_unsqueezed = x.unsqueeze(-1).to("cuda", torch.float)  # (N, D, 1)
        x_unsqueezed = x.unsqueeze(-1).to(torch.float)  # (N, D, 1)
        scaled = (
            self.frequencies.view(1, 1, -1) * x_unsqueezed
        )  # (N, D, num_frequencies)
        s = torch.sin(scaled)
        c = torch.cos(scaled)
        embedded = torch.cat([s, c, x_unsqueezed], dim=-1).view(
            N, -1
        )  # (N, D * 2 * num_frequencies + D)
        return embedded


class Transformer(nn.Module):

    """
    The G.pt model.
    """
### For GPT model:
# n_embd: 2880  ### The channel dim
# n_layer: 12   ### n layers in each block
# n_head: 16    ### n head in attention layer
# split_policy: layer_by_layer ### Each layer is one token

### For Transformer init:
# use_global_residual: False ### No global residual connection. They did implement it in the code but not used. Maybe it was originally for the G.pt paper.
# condition: 'no' ### Can't find the usage of this variable in the code. WeightDataset define this variable but not used either.
    def __init__(
        self,
        parameter_sizes,  # A list of integers indicating the total number of parameters in each layer ### [3456, 128, 16384, 128, 16384, 128, 128, 1]
        parameter_names,  # A list of strings indicating the name of each layer in the input networks ### ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias', 'layers.2.weight', 'layers.2.bias', 'layers.3.weight', 'layers.3.bias']
        num_frequencies=128,  # number of frequencies sampled for embedding scalars
        max_freq_log2=20,  # max log2 frequency for embedding scalars
        predict_xstart=True,  # if True, G.pt predicts signal (False = predict noise)
        absolute_loss_conditioning=False,  # if True, adds two extra input tokens indicating starting/target metrics
        use_global_residual=False,
        condition="no",
        condition_n_points=0,
        **gpt_kwargs,  # Arguments for the Transformer model (depth, heads, etc.)
    ):
        super().__init__()
        self.predict_xstart = predict_xstart
        self.absolute_loss_conditioning = absolute_loss_conditioning
        self.dims = 0  # This is for compatibility with UNet
        self.ae_model = None  # This is for compatibility with UNet
        self.condition = condition
        self.condition_n_points = condition_n_points
        self.use_global_residual = use_global_residual

        ### Compute the token sizes
        ### Based on the weight's sizes and names.
        ### input_parameter_sizes: [[3456, 128, 16384, 128, 16384, 128, 128, 1], [257]]
        ### output_parameter_sizes: [[3456, 128, 16384, 128, 16384, 128, 128, 1]]
        ### input_parameter_names: ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias', 'layers.2.weight', 'layers.2.bias', 'layers.3.weight', 'layers.3.bias', 'timestep_embedding']
        (
            input_parameter_sizes,  ### parameter_sizes + other parameters (e.g., timestep, embedding, etc.)
            output_parameter_sizes, ### parameter_sizes (The mlps' weights)
            input_parameter_names,  ### parameter_names + other parameters names (e.g., timestep, embedding, etc.)
        ) = self.compute_token_sizes(parameter_sizes, parameter_names, num_frequencies)

        # print(f"\nFrom {self.__class__.__name__}")
        # print(f"input_parameter_sizes: {input_parameter_sizes} ; total: {sum([sum(x) for x in input_parameter_sizes])}") ### [[3456, 128, 16384, 128, 16384, 128, 128, 1], [257]]
        # print(f"output_parameter_sizes: {output_parameter_sizes} ; total: {sum([sum(x) for x in output_parameter_sizes])}") ### [[3456, 128, 16384, 128, 16384, 128, 128, 1]]
        # print(f"input_parameter_names: {input_parameter_names}") ### ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias', 'layers.2.weight', 'layers.2.bias', 'layers.3.weight', 'layers.3.bias', 'timestep_embedding']

        self.input_parameter_sizes = input_parameter_sizes

        ### Create the GPT model
        self.decoder = GPT(
            input_parameter_sizes,
            output_parameter_sizes,
            input_parameter_names,
            **gpt_kwargs,
        )

        ### Timestep embedding
        self.scalar_embedder = FrequencyEmbedder(num_frequencies, max_freq_log2)

        # Initialize with identity output:
        if self.use_global_residual:
            for out_proj in self.decoder.output_parameter_projections:
                out_proj[-1].weight.data.zero_()

    @staticmethod
    def get_scalar_token_size(num_frequencies):
        """
        Computes the size of each metadata token after being projected by the frequency embedder.
        """
        return num_frequencies * 2 + 1

    def compute_token_sizes(self, parameter_sizes, parameter_names, num_frequencies):
        """
        This function returns a few different lists which are used to construct the GPT model.

        input_parameter_sizes: A list that breaks-down the sizes of the different input vectors.
        output_parameter_sizes: A list that breaks-down the sizes of the different vectors the G.pt model will output.
        input_parameter_names: A list that contains string names for every individual input layer and scalar.

        For example, say we have a linear network with a (10, 784)-shape weight and a (10,)-shape bias, and we embed
        each input scalar into a 257-dimensional vector. Then this function might return the following:

        input_parameter_sizes: [[7840, 10], [7840, 10], [257], [257], [257], [257]]
        output_parameter_sizes: [[7840, 10]]  # G.pt only outputs denoised parameters
        input_parameter_names: ['weight', 'bias', 'weight', 'bias', 'timestep_embedding',
                               'loss_delta_embedding', 'target_loss_embedding', 'current_loss_embedding']

        These lists are used by the GPT class above to determine how to split the input vector into different tokens.
        """
        ### parameter_sizes: [3456, 128, 16384, 128, 16384, 128, 128, 1]
        ### parameter_names: ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias', 'layers.2.weight', 'layers.2.bias', 'layers.3.weight', 'layers.3.bias']
        ### num_frequencies: 128

        ### This function will create lists for input_parameter_sizes, output_parameter_sizes, input_parameter_names:
        ### input_parameter_sizes: [[3456, 128, 16384, 128, 16384, 128, 128, 1], [257]]
        ### output_parameter_sizes: [[3456, 128, 16384, 128, 16384, 128, 128, 1]]
        ### input_parameter_names: ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias', 'layers.2.weight', 'layers.2.bias', 'layers.3.weight', 'layers.3.bias', 'timestep_embedding']
        
        input_parameter_sizes = [deepcopy(parameter_sizes)]
        output_parameter_sizes = [deepcopy(parameter_sizes)] ### Basically the same as parameter_sizes
        input_parameter_names = deepcopy(parameter_names)

        # account for the second weight vector that will be input:
        if self.use_global_residual:
            input_parameter_sizes.append(input_parameter_sizes[0])
            input_parameter_names.extend(input_parameter_names)

        ### Calculate the size of the timestep embedding to add to the input
        ### The scalar_token_size is the size of the timestep embedding = num_frequencies * 2 + 1 = 128 * 2 + 1 = 257
        # Account for the scalar inputs (diffusion timestep and loss/error/return inputs):
        scalar_token_size = [self.get_scalar_token_size(num_frequencies)]
        input_parameter_sizes.extend([scalar_token_size])
        input_parameter_names.extend(["timestep_embedding"])

        return input_parameter_sizes, output_parameter_sizes, input_parameter_names

    def configure_optimizers(self, lr, wd, betas):
        """
        Sets up the AdamW optimizer for G.pt (no weight decay on the positional embeddings or layer norm biases).
        """
        return GPT.configure_optimizers(self, lr, wd, betas)

    ### Have no idea what this function is for because it is not used in the code.
    @torch.no_grad()
    def gradient_norm(self):
        """
        Computes the gradient norm for monitoring purposes.
        """
        total_norm = 0.0
        for p in self.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    ### The forward pass of the Transformer
    def forward(self, x, t, x_prev=None):
        """
        Full G.pt forward pass.
        ----------------------------------------------
        N = batch size
        D = number of parameters
        ----------------------------------------------
        x: (N, D) tensor of noised updated parameters
        t: (N, 1) tensor indicating the diffusion timestep
        loss_target: (N, 1) tensor, the prompted (desired) loss/error/return
        loss_prev: (N, 1) tensor, loss/error/return obtained by x_prev
        x_prev: (N, D) tensor of starting parameters that are being updated
        ----------------------------------------------
        returns: (N, D) tensor of denoised updated parameters
        ----------------------------------------------
        """
        # print(f"\nFrom {self.__class__.__name__}")
        # print(f"x.shape: {x.shape}") ### [32, 36737] (training batch size = 32, while test batch size = 100)

        t_embedding = self.scalar_embedder(t)
        # print(f"t_embedding.shape: {t_embedding.shape}") ### [32, 257]

        # loss_embedding = self.embed_loss(loss_target, loss_prev)

        ### self.use_global_residual=False in config file
        if self.use_global_residual:
            x_prev = x_prev.unsqueeze(0).repeat((len(x), 1))
            assert x.shape == x_prev.shape
            inp = [x, x_prev, t_embedding]

        else: ### Will enter this branch
            inp = [x, t_embedding]
        

        ### Concate x and t_embedding along the channel dimension
        ### inp.shape: [B, n_weight + timestep_embedding]
        inp = torch.cat(inp, 1)
        # print(f"After `inp = torch.cat(inp, 1)`, inp.shape: {inp.shape}") ### [32, 36994]

        output = self.decoder(inp)
        # print(f"output.shape: {output.shape}") ### [32, 36737]

        ### self.use_global_residual=False in config file
        # TODO: Global residual connection:
        if self.use_global_residual:
            output = output + x_prev

        return output


if __name__ == "__main__":
    from mlp_models import MLP

    ### Added by Khoa
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda")
    print(f"Run code in {device}")

    ### Create MLP
    mlp = MLP(
        in_size=6,
        out_size=1,
        hidden_neurons=[16, 16, 16],
        use_tanh=True,
        over_param=False,
    )
    print(f"\nmlp: {mlp}")

    ### Get weights
    state_dict = mlp.state_dict()
    print(f"\nstate_dict: {state_dict}")

    ### Create weights input
    layers = []
    layer_names = []
    input = []

    for l in state_dict:
        shape = state_dict[l].shape
        layers.append(np.prod(shape))
        layer_names.append(l)
        input.append(state_dict[l].flatten())
    input = torch.hstack(input).unsqueeze(0).to(device)
    print(f"\nlayers: {layers}")
    print(f"layer_names: {layer_names}")

    ### Create Transformer
    net = Transformer(layers, layer_names, split_policy="layer_by_layer").to(device)

    ### Random timestep
    t = torch.randint(0, 1000, (len(input), 1)).to(device)
    print(f"\nInput and timestep shape: {input.shape}, {t.shape}")

    out = net(input, t)
    print(f"\nOutput shape: {out.shape}")