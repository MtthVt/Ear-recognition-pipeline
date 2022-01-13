import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from detectors.unet_segmentation.unet_attention.networks.unet_grid_attention_2D import UNet_Attention
from metrics.evaluation import Evaluation
from utils.data_loading import BasicDatasetDetection, TransformDatasetDetection
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

dir_img_train = Path('../../data/ears/train/')
dir_img_test = Path('../../data/ears/test/')
dir_mask = Path('../../data/ears/annotations/segmentation/train')
dir_mask_test = Path('../../data/ears/annotations/segmentation/test')
dir_checkpoint = Path('./checkpoints/')


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              amp: bool = False):
    # 1. Create dataset
    dataset = TransformDatasetDetection(dir_img_train, dir_mask, length=750, scale=img_scale)

    #### Calculation of data distribution
    # dataset = BasicDataset(dir_img_test, dir_mask_test, scale=img_scale)
    # loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    # train_loader = DataLoader(dataset, shuffle=False, **loader_args)
    # dist = {0: 0, 1: 0}
    # for batch in train_loader:
    #     true_masks = batch['mask']
    #     unique, counts = np.unique(true_masks, return_counts=True)
    #     tmp = dict(zip(unique, counts))
    #     dist[0] += tmp[0]
    #     dist[1] += tmp[1]
    # print(dist[0] / dist[1])

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    betas = (0.9, 0.999)
    eps = 1e-08
    cross_entropy_weight = [1., 5.]  # 2nd class (ear) is way rarer -> adapt loss function
    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint,
                                  amp=amp, optimizer='ADAM', betas=betas, eps=eps,
                                  cross_entropy_weight=cross_entropy_weight, architecture="UNET",
                                  augmentation="Transform_less_noaffine"))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, amsgrad=True,
                           betas=betas,
                           eps=eps,
                           weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=4)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_weights = torch.tensor(cross_entropy_weight)
    loss_weights = torch.FloatTensor(loss_weights).to(device=device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (2 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        # scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            # 'images': wandb.Image(images[0].cpu()),
                            # 'masks': {
                            #     'true': wandb.Image(true_masks[0].cpu()),
                            #     'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            # },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            run_name = experiment.name
            torch.save(net.state_dict(), str(dir_checkpoint / '{}_cp_epoch{}.pth'.format(run_name, epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')
    return experiment


def test_net(net, device, experiment):
    iou_arr = []
    # preprocess = Preprocess()
    eval = Evaluation()

    # 1. Create dataset
    test_set = BasicDatasetDetection(dir_img_test, dir_mask_test, 1.0)

    # 2. Create data loader
    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    # load unet model if not given as parameter
    if net is None:
        # UNET Segmentation network
        checkpoint = "detectors/unet_segmentation/checkpoints/checkpoint_epoch5.pth"
        net = UNet(n_channels=3, n_classes=2, bilinear=True)
        net.load_state_dict(torch.load(checkpoint, map_location=device))
        net.to(device=device)

    counter = 0
    dice_score = 0
    for batch in test_loader:
        images = batch['image']
        mask_true = batch['mask']
        images = images.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        # Convert to one-hot
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        # Get the UNET result
        mask_pred = net.forward(images)
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

        # Compare to true mask

        mask_pred_np = mask_pred.cpu().detach().numpy()
        mask_true_np = mask_true.cpu().detach().numpy()
        iou = eval.iou_compute(mask_pred_np[0][1], mask_true_np[0][1])
        iou_arr.append(iou)
        # compute the Dice score, ignoring background
        # dice_score += multiclass_dice_coeff(mask_pred_np[:, 1:, ...], mask_true_np[:, 1:, ...], reduce_batch_first=False)
        counter += 1
        # print(counter)
        # print(iou)

    miou = np.average(iou_arr)
    # dice_score /= counter
    print("\n")
    print("Average IOU:", f"{miou:.2%}")
    # print("Average Dice score:", f"{dice_score:.2%}")
    print("\n")
    if experiment is not None:
        experiment.log({
            'test miou': miou
        })


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=5, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    # checkpoint = "checkpoints/revived-night-66_cp_epoch5.pth"
    # net.load_state_dict(torch.load(checkpoint, map_location=device))
    # net = UNet_Attention(n_channels=3, n_classes=2, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        experiment = train_net(net=net,
                               epochs=args.epochs,
                               batch_size=args.batch_size,
                               learning_rate=args.lr,
                               device=device,
                               img_scale=args.scale,
                               val_percent=args.val / 100,
                               amp=args.amp)
        # for i in range(20,29):
        #     experiment = None
        #     run_name = "rose-wildflower-76"
        #     checkpoint = str(dir_checkpoint / '{}_cp_epoch{}.pth'.format(run_name, i))
        #     print("Evaluate " + checkpoint)
        #     net.load_state_dict(torch.load(checkpoint, map_location=device))
        test_net(net=net,
                 device=device,
                 experiment=experiment)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
