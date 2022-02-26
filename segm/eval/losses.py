import torch
import numpy as np


def hard_worst_loss(loss, groundtruth):
    """_summary_

    Args:
        loss (_type_): _description_
        groundtruth (_type_): _description_

    Returns:
        _type_: _description_
    """
    b, h, w = loss.size()

    # Flatten tensor
    loss = loss.contiguous().view(b * h * w)
    groundtruth = groundtruth.contiguous().view(b * h * w)

    foreground_idx = groundtruth.nonzero().nonzero().permute(1, 0)[0]
    background_idx = (groundtruth == 0).nonzero().permute(1, 0)[0]

    fore_loss = loss[foreground_idx]
    back_loss = loss[background_idx]

    n_back_samples = min(foreground_idx.size(0) * 3, background_idx.size(0))
    top_k_back_loss_idx = np.argsort(-back_loss.data.cpu().numpy())[:n_back_samples]

    back_loss = back_loss[top_k_back_loss_idx]

    print(fore_loss.mean(), back_loss.mean())

    return 3 * fore_loss.mean() + back_loss.mean()

    # return (fore_loss.sum() + back_loss.sum()) / (fore_loss.size(0) + back_loss.size(0))