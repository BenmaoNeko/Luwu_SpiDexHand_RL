import pytest
import torch
from wuji_mjlab.rl.bottom_tip_ppo import retention_gradient_scale


def test_large_finite_gradients_do_not_make_retention_scale_nan():
    ppo = [torch.tensor([1e22, -1e22])]
    retained = [torch.tensor([2e22, -2e22])]
    scale = retention_gradient_scale(ppo, retained)
    assert torch.isfinite(scale)
    assert scale.item() == pytest.approx(.1)


def test_retention_scale_caps_at_one_and_handles_unused_parameters():
    assert retention_gradient_scale([None, torch.ones(2)], [None, torch.zeros(2)]) == 1


def test_nonfinite_gradient_fails_before_optimizer_can_corrupt_weights():
    with pytest.raises(FloatingPointError, match='gradient'):
        retention_gradient_scale([torch.tensor([float('nan')])], [torch.ones(1)])
