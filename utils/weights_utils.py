"""
This file contains some useful functions for deep weight space.

Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Disclaimer: This file is taken and modified from many sources.
Updated: 2025-02-08
"""

import torch


### Take and modified from HyperDiffusion
def state_dict_to_weights(state_dict : dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Convert a state_dict to a weights tensor by flattening each weight and stacking them horizontally.

    Best practice: Appending each flattened weight to the list and then stack them horizontally is much faster 
    than creating a placeholder tensor and concatenating them immediately in each iteration of the for-loop.
    """
    weights = []

    ### Read each weight, flatten it and append to the list
    for w in state_dict.values():
        weights.append(w.flatten())

    ### Stack the weights into a single tensor
    ### In this case, concatenation along the first axis for 1-D tensors.
    weights = torch.hstack(weights) ### (n_weights, )

    return weights


def add_weights_to_mlp(mlp_model: torch.nn.Module, weights_vector: torch.Tensor) -> torch.nn.Module:
    """
    Assign flattened weights to the MLP parameters with shape matching.

    Note: weights_vector here is a flattened vector [n_weights,], which means all the weights are stacked horizontally. It's not a state_dict.

    Args:
        mlp_model (torch.nn.Module): The MLP model to assign the weights to.
        weights_vector (torch.Tensor): The flattened weights vector.

    Returns:
        torch.nn.Module: The same MLP model with the assigned weights.
    """
    state_dict = mlp_model.state_dict()

    ### Assign the weights to the parameters
    start_idx = 0
    for name, param in mlp_model.named_parameters():
        shape = param.shape
        length = param.numel()
        end_idx = start_idx + length
        state_dict[name] = weights_vector[start_idx:end_idx].view(shape)
        start_idx = end_idx

    ### Load the state_dict to the model
    mlp_model.load_state_dict(state_dict)

    return mlp_model
