import torch

from crl.utils.count_params import count_params


def test_count_params_ignores_frozen_parameters():
    module = torch.nn.Linear(3, 2)
    module.bias.requires_grad = False

    assert count_params(module) == 6
