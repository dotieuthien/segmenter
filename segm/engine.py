import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu
from segm.eval.losses import hard_worst_loss

from torchvision import transforms


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
):
    weights = torch.tensor(data_loader.unwrapped.weighted_loss).float().to(ptu.device)
    # weights = np.array([1, 10, 2, 10, 10, 10, 10])
    # weights = weights / np.sum(weights)
    # weights = torch.tensor(weights).float().to(ptu.device)
    # criterion1 = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=0)
    criterion2 = torch.nn.CrossEntropyLoss(reduce=False, weight=weights)

    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)

    total_loss = 0

    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)

        # with amp_autocast():
        seg_pred = model.forward(im)

        # loss1 = 10 * criterion1(seg_pred, seg_gt).mean()

        loss2 = criterion2(seg_pred, seg_gt).mean()
        # loss2 = hard_worst_loss(loss2, seg_gt)

        loss = loss2

        # Convert input tensor to image
        func = transforms.ToPILImage()
        im_rgb = func(im[0])

        # Convert prediction to image
        seg_pred_img = seg_pred[0]
        seg_pred_img = seg_pred_img.argmax(0, keepdim=True)
        seg_pred_img = seg_pred_img.cpu().detach().numpy()[0]
        
        # Convert groundtruth to image
        seg_gt_img = seg_gt.cpu().detach().numpy()[0]

        print('Number of value in prediction ', np.unique(seg_pred_img))
        print('Number of value in gt ', np.unique(seg_gt_img))


        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()

        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        logger.update(
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )

        total_loss += loss_value

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(im_rgb)
    fig.add_subplot(1, 3, 2)
    plt.imshow(seg_pred_img)
    fig.add_subplot(1, 3, 3)
    plt.imshow(seg_gt_img)
    # plt.show()

    neptune_stats = {'loss': total_loss / len(data_loader), 'segmap': seg_pred_img, 'gtmap': seg_gt_img, 'fig': fig}
    return logger, neptune_stats


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    val_seg_pred = {}
    model.eval()

    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                ims,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
            )
            seg_pred = seg_pred.argmax(0)

        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename] = seg_pred

    val_seg_pred = gather_data(val_seg_pred)
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        data_loader.unwrapped.n_cls,
        ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger
