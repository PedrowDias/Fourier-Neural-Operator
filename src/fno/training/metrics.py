import torch


def relative_l2_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''Relative L2 error, averaged over the batch.

    This is the standard evaluation metric for neural operators, as used in
    Li et al. (2020). It normalises by the L2 norm of the target so errors
    are comparable across initial conditions with different magnitudes.

    Args:
        prediction: Shape (batch, ...).
        target:     Shape (batch, ...).

    Returns:
        Scalar tensor — mean relative error across the batch.
    '''
    error_norm = torch.norm(prediction - target, dim=-1)
    target_norm = torch.norm(target, dim=-1)
    return torch.mean(error_norm / (target_norm + 1e-8))


def absolute_l2_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''Mean absolute L2 error over the batch.

    Useful as a secondary metric when comparing solvers on a fixed dataset
    where all targets have similar magnitude.

    Args:
        prediction: Shape (batch, ...).
        target:     Shape (batch, ...).

    Returns:
        Scalar tensor.
    '''
    return torch.mean(torch.norm(prediction - target, dim=-1))
