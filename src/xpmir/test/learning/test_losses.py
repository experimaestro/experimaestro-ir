import torch
from torch.autograd import gradcheck
from xpmir.learning.losses import bce_with_logits_loss


def test_pairwise_bce_loss():
    input = -torch.randn(
        (12,), dtype=torch.double, requires_grad=True
    ).abs(), torch.randint(0, 2, (12,))
    gradcheck(bce_with_logits_loss, input, eps=1e-6, atol=1e-4)
