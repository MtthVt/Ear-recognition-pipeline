import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches


def evaluate_recognition(net, dataloader, device):
    """
    Evaluate a recognition model via calculating it's mean accuracy.
    :param net: Network to evaluate
    :param dataloader:  Dataloader for the evaluation data
    :param device:  CUDA device to perform the computations on
    :return: mean accuracy
    """
    net.eval()      # Important for pytorch internals!
    num_val_batches = len(dataloader)
    batch_size = dataloader.batch_size

    acc_count = 0.
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, id_true = batch['image'], batch['id']
        # move images and ids to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        id_true = id_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the ids
            id_pred = net(image)

            # Get the resulting ids from the prediction
            id_pred = id_pred.argmax(dim=1).long()

            # Calculate the number of matching ids (predict/true)
            acc_count += torch.sum(torch.eq(id_pred, id_true)).item()
    # Put the network back into train mode
    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        # return ce_loss
        return acc_count
    # return ce_loss / num_val_batches
    return acc_count / (num_val_batches*batch_size)
