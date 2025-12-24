
# from omegaconf import DictConfig

import torch
from torch import nn

from einops.layers.torch import Rearrange, Reduce

from diffusers import UNet1DModel
from denoising_diffusion_pytorch import Unet1D as LucidrainsUNet1D




class mlp_proj_layer(nn.Module):
    def __init__(self,
                 parameter_sizes: list[int],
                 n_embd: int = 1024,
                 mlp_proj_encoder_depth: int = 1,
                 mlp_proj_decoder_depth: int = 1,
                 ):
        super().__init__()
        self.parameter_sizes = parameter_sizes
        self.parameter_length = len(parameter_sizes)
        self.n_weights = sum(parameter_sizes)
        self.n_embd = n_embd
        self.mlp_proj_encoder_depth = mlp_proj_encoder_depth
        self.mlp_proj_decoder_depth = mlp_proj_decoder_depth

        ### Create the projection layer for each token
        ### # (B, C) --MLP--> (B, parameter_length, n_embd)
        self.input_parameter_projections = self.build_encoder()
        self.mlp_proj_norm = nn.LayerNorm(self.n_embd)
        self.output_parameter_projections = self.build_decoder()
    
    def build_encoder(self):
        ### Create a unique MLP encoder for each token
        input_parameter_projections = nn.ModuleList()
        
        ### self.parameter_sizes = [3456, 128, 16384, 128, 16384, 128, 128, 1]
        for param_chunk_size in self.parameter_sizes:
            in_proj = [nn.Linear(param_chunk_size, self.n_embd, bias=False)]
            for _ in range(self.mlp_proj_encoder_depth - 1):
                in_proj.append(nn.GELU())
                in_proj.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
            in_proj = nn.Sequential(*in_proj)
            input_parameter_projections.append(in_proj)

        return input_parameter_projections

    def build_decoder(self):
        # Create a unique MLP decoder for each noised token
        output_parameter_projections = nn.ModuleList()

        for output_chunk_size in self.parameter_sizes:
            out_proj = []

            ### mlp_proj_decoder_depth: 1 in train_plane.yaml
            ### It means this for loop will not run
            for _ in range(self.mlp_proj_decoder_depth - 1): 
                out_proj.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
                out_proj.append(nn.GELU())

            ### Create MLP for each token (Weights or Biases, etc individually)
            out_proj.append(nn.Linear(self.n_embd, output_chunk_size, bias=False))
            out_proj = nn.Sequential(*out_proj)
            output_parameter_projections.append(out_proj)

        return output_parameter_projections

    def encode_parameters(self, parameters, output_order="BLC"):
        assert output_order in ["BLC", "BCL"]
        split_parameters = torch.split(parameters, self.parameter_sizes, dim=1)

        representations = []
        for parameter, in_proj in zip(split_parameters, self.input_parameter_projections):
            ### Apply individual MLP to each chunk (Weights or Biases, etc.)
            representations.append(in_proj(parameter))

        representations = torch.stack(representations, dim=1)  # (B, L, C)
        representations = self.mlp_proj_norm(representations)
        if output_order == "BCL":
            representations = representations.permute(0, 2, 1) ### (B, C, L)
        assert representations.dim() == 3  
        return representations ### [B, token length, n_embd] ### [32, 8, n_embd]

    def decode_parameters(self, features, input_order="BLC"):
        assert features.dim() == 3  # (B, L, C) ### [B, token length, n_embd] ### [32, 9, 2880]
        assert input_order in ["BCL", "BLC"]
        if input_order == "BCL":
            features = features.permute(0, 2, 1) ### (B, C, L)

        output = []

        for t in range(len(self.parameter_sizes)):
            ### Get the MLP corresponding to the t-th token
            out_proj = self.output_parameter_projections[t]

            ### Apply the MLP to the t-th token
            output.append(out_proj(features[:, t, :]))

        ### Concatenate the output of all the MLPs
        ### output is a list of [B, output_chunk_size]
        output = torch.cat(output, 1)  # (b, c)
        assert output.dim() == 2

        return output



class UNet1DDiffusion(nn.Module):
    def __init__(self, 
                 parameter_sizes: list[int],
                 n_embd: int = 1024,
                 length_size: int = 32, ### should be >= 2^(len(block_out_channels) + 2)
                 use_mlp_proj: bool = True,
                 mlp_proj_encoder_depth: int = 1, ### Depth of the mlp_proj
                 mlp_proj_decoder_depth: int = 1,
                 use_up_proj_expand: bool = False,
                 down_block_types: tuple[str, ...] = ("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
                 up_block_types: tuple[str, ...] = ("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
                 mid_block_type: str = "UNetMidBlock1D",
                 block_out_channels: tuple[int, ...] = (64, 128, 256),
                 ):
        super().__init__()
        self.parameter_sizes = parameter_sizes
        self.n_weights = sum(parameter_sizes)
        self.parameter_length = len(parameter_sizes)
        self.n_embd = n_embd
        self.length_size = length_size
        self.use_mlp_proj = use_mlp_proj
        self.mlp_proj_encoder_depth = mlp_proj_encoder_depth
        self.mlp_proj_decoder_depth = mlp_proj_decoder_depth
        self.use_up_proj_expand = use_up_proj_expand

        if self.use_mlp_proj:
            ### Create the projection layer for each token
            ### # (B, C) --MLP--> (B, parameter_length, n_embd)
            self.input_parameter_projections = self.build_encoder()
            self.mlp_proj_norm = nn.LayerNorm(self.n_embd)
            self.output_parameter_projections = self.build_decoder()
            
            ### (B, L, C) --> (B, C, L) --> (B, C, length_size)
            self.up_proj = nn.Sequential(
                Rearrange("b l c -> b c l"),
                nn.Linear(self.parameter_length, self.length_size, bias=False),
            )
            self.down_proj = nn.Sequential(
                nn.Linear(self.length_size, self.parameter_length, bias=False),
                Rearrange("b c l -> b l c"),
            )

            in_channels = self.n_embd

        else:
            if self.use_up_proj_expand:
                self.up_proj = Rearrange("b c -> b c 1")
                self.down_proj = Reduce("b c l -> b c", "mean")

            else:
                ### (B, C) --> (B, C, length_size)
                self.up_proj = nn.Sequential(
                    Rearrange("b c -> b c 1"),
                    nn.Linear(1, self.length_size, bias=False),
                )
                self.down_proj = nn.Sequential(
                    nn.Linear(self.length_size, 1, bias=False),
                    Rearrange("b c 1 -> b c"),
                )

            in_channels = self.n_weights

        ### Define a simple 1D UNet model
        self.unet = UNet1DModel(
            sample_size=self.length_size,
            in_channels=in_channels,
            out_channels=in_channels,
            extra_in_channels=16,
            time_embedding_type="fourier",
            flip_sin_to_cos=True, ### [sin, cos] if true, [cos, sin] if false
            use_timestep_embedding=False, ### If True, time_embed_dim = block_out_channels[0] * 4. Calculate the in_channels again.
            freq_shift=0.0, ### Only use for time_embedding_type="positional"
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            mid_block_type=mid_block_type,
            out_block_type=None,
            block_out_channels=block_out_channels,
            act_fn=None,
            norm_num_groups=8,
            layers_per_block=1, ### Only use for DownResnetBlock1D, UpResnetBlock1D
            downsample_each_block=False,
        )

    def build_encoder(self):
        ### Create a unique MLP encoder for each token
        input_parameter_projections = nn.ModuleList()
        
        ### self.parameter_sizes = [3456, 128, 16384, 128, 16384, 128, 128, 1]
        for param_chunk_size in self.parameter_sizes:
            in_proj = [nn.Linear(param_chunk_size, self.n_embd, bias=False)]
            for _ in range(self.mlp_proj_encoder_depth - 1):
                in_proj.append(nn.GELU())
                in_proj.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
            in_proj = nn.Sequential(*in_proj)
            input_parameter_projections.append(in_proj)

        return input_parameter_projections

    def build_decoder(self):
        # Create a unique MLP decoder for each noised token
        output_parameter_projections = nn.ModuleList()

        for output_chunk_size in self.parameter_sizes:
            out_proj = []

            ### mlp_proj_decoder_depth: 1 in train_plane.yaml
            ### It means this for loop will not run
            for _ in range(self.mlp_proj_decoder_depth - 1): 
                out_proj.append(nn.Linear(self.n_embd, self.n_embd, bias=False))
                out_proj.append(nn.GELU())

            ### Create MLP for each token (Weights or Biases, etc individually)
            out_proj.append(nn.Linear(self.n_embd, output_chunk_size, bias=False))
            out_proj = nn.Sequential(*out_proj)
            output_parameter_projections.append(out_proj)

        return output_parameter_projections

    def encode_parameters(self, parameters):
        split_parameters = torch.split(parameters, self.parameter_sizes, dim=1)

        representations = []
        for parameter, in_proj in zip(split_parameters, self.input_parameter_projections):
            ### Apply individual MLP to each chunk (Weights or Biases, etc.)
            representations.append(in_proj(parameter))

        representations = torch.stack(representations, dim=1)  # (B, L, C)
        representations = self.mlp_proj_norm(representations)
        assert representations.dim() == 3  
        return representations ### [B, token length, n_embd] ### [32, 8, n_embd]

    def decode_parameters(self, features):
        assert features.dim() == 3  # (B, L, C) ### [B, token length, n_embd] ### [32, 9, 2880]
        output = []

        for t in range(len(self.parameter_sizes)):
            ### Get the MLP corresponding to the t-th token
            out_proj = self.output_parameter_projections[t]

            ### Apply the MLP to the t-th token
            output.append(out_proj(features[:, t, :]))

        ### Concatenate the output of all the MLPs
        ### output is a list of [B, output_chunk_size]
        output = torch.cat(output, 1)  # (b, c)
        assert output.dim() == 2

        return output
    

    def forward(self, x, t):
        ### x is a tensor of shape (B, C)
        ### t is a tensor of shape (B,)

        ### Encode the parameters
        if self.use_mlp_proj:
            x = self.encode_parameters(x) ### (B, L, C)

        x = self.up_proj(x) ### (B, C, L)
        if not self.use_mlp_proj and self.use_up_proj_expand:
            x = x.broadcast_to(-1, -1, self.length_size) ### (B, C, 1) --> (B, C, length_size)

        ### Pass through the Transformer
        x = self.unet(x, t).sample ### (B, C, L)

        x = self.down_proj(x) ### (B, L, C)

        ### Decode the parameters
        if self.use_mlp_proj:
            x = self.decode_parameters(x) ### (B, C)

        return x


class LucidrainsUNet1DDiffusion(mlp_proj_layer):
    def __init__(self, 
                 parameter_sizes: list[int],
                 n_embd: int = 1024,
                 mlp_proj_encoder_depth: int = 1,
                 mlp_proj_decoder_depth: int = 1,
                 length_size: int | None = None,
                 unet_dim: int = 64,
                 unet_dim_mults: tuple[int, ...] = (1, 2, 4, 8),
                 ):
        ### Initialize the mlp_proj_layer
        super().__init__(parameter_sizes, n_embd, mlp_proj_encoder_depth, mlp_proj_decoder_depth)
        
        # Add positional embedding for sequence dimension
        self.pos_emb = nn.Parameter(torch.zeros(1, n_embd, self.parameter_length))
        
        self.length_size = self.parameter_length
        if length_size is not None:
            self.length_size = length_size
            # self.up_proj = nn.Sequential(
            #     nn.GELU(),
            #     nn.Linear(self.parameter_length, self.length_size, bias=False),
            #     nn.LayerNorm(self.length_size),
            # )
            # self.down_proj = nn.Sequential(
            #     nn.GELU(),
            #     nn.Linear(self.length_size, self.parameter_length, bias=True),
            #     nn.LayerNorm(self.parameter_length)
            # )
            self.up_proj = nn.Sequential(
                nn.Linear(self.parameter_length, self.length_size, bias=True),
                # nn.LayerNorm(self.length_size),
                # nn.GELU(),
            )
            self.down_proj = nn.Sequential(
                nn.Linear(self.length_size, self.parameter_length, bias=True),
                # nn.LayerNorm(self.parameter_length),
                # nn.GELU(),
            )


        ### Define a simple 1D UNet model
        self.unet = LucidrainsUNet1D(
            dim = unet_dim,
            dim_mults = unet_dim_mults,
            channels = n_embd ### n_embd is the number of channels in the input
        )
        ### Assigning these values for GaussianDiffusion1D
        self.channels = self.unet.channels
        self.self_condition = self.unet.self_condition

    def forward(self, x, t, *args, **kwargs):
        ### x is a tensor of shape (B, C)
        ### t is a tensor of shape (B,)

        ### Encode the parameters
        x = self.encode_parameters(x, output_order="BCL") ### (B, C, L)

        ### Add positional embedding for sequence dimension
        x = x + self.pos_emb

        if self.length_size != self.parameter_length:
            x = self.up_proj(x) ### (B, C, L)

        ### Pass through the Transformer
        x = self.unet(x, t, *args, **kwargs) ### (B, C, L)

        if self.length_size != self.parameter_length:
            x = self.down_proj(x) ### (B, C, L)

        x = self.decode_parameters(x, input_order="BCL") ### (B, C)

        return x






if __name__ == "__main__":
    parameter_sizes = [3456, 128, 16384, 128, 16384, 128, 128, 1]
    batch_size = 2
    length_size = 32 ### should be >= 2^(len(block_out_channels) + 2)
    # unet = UNet1DDiffusion(parameter_sizes=parameter_sizes, length_size=length_size)

    unet = LucidrainsUNet1DDiffusion(parameter_sizes=parameter_sizes, length_size=length_size)

    x = torch.randn(batch_size, sum(parameter_sizes))
    t = torch.randint(0, 1000, (batch_size,))
    print(unet(x, t).shape)
