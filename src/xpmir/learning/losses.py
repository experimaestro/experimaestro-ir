import torch


class BCEWithLogLoss(torch.autograd.Function):
    """Binary cross-entropy loss when outputs are log probabilities

    Both arguments should be vectors

    First argument: log_probs must be strictly negative
    Second arguments: targets, can be betwen 0 and 1
    """

    @staticmethod
    def forward(ctx, log_probs: torch.Tensor, targets: torch.Tensor):
        assert torch.all(log_probs < 0.0)

        # Computes the loss
        p_neg = 1.0 - log_probs.exp()
        loss = (
            -(targets * log_probs).sum()
            - ((1 - targets) * (1.0 - log_probs.exp()).log()).sum()
        )

        # Save for backward
        ctx.save_for_backward(log_probs, p_neg, targets)

        return loss / log_probs.numel()

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None

        log_probs, p_neg, targets = ctx.saved_tensors

        # Stabilizes the gradient (see Loss.cpp in Aten)
        p_neg = p_neg.clamp_min(1e-12)
        grad_output = grad_output / log_probs.numel()

        grad_log_probs = -targets + (targets - 1) * (1.0 - 1.0 / p_neg)
        return grad_output * grad_log_probs, None


bce_with_logits_loss = BCEWithLogLoss.apply
