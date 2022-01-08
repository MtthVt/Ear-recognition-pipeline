import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate_recognition
from feature_extractors.resnet.resnet_model import ResNet
from utils.data_loading import BasicDatasetRecognition

dir_img_train = Path('../data/perfectly_detected_ears/train/')
dir_img_test = Path('../data/perfectly_detected_ears/test/')
dict_id_translation = Path('../data/perfectly_detected_ears/annotations/recognition/ids.csv')
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
    dataset = BasicDatasetRecognition(dir_img_train, dict_id_translation, net.n_classes)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Adam parameters
    betas = (0.9, 0.999)
    eps = 1e-08

    # (Initialize logging)
    experiment = wandb.init(project='Ear-Recognition', resume='allow', entity='min0x')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint,
                                  amp=amp, optimizer='ADAM', betas=betas, eps=eps,
                                  architecture="Resnet-own", augmentation="grayscale"))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, amsgrad=True,
                           betas=betas,
                           eps=eps,
                           weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0

        # Iterate over the whole training dataset
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # Load the images & ids from the batch, transfer to CUDA device.
                images = batch['image']
                true_ids = batch['id']

                images = images.to(device=device, dtype=torch.float32)
                true_ids = true_ids.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=amp):
                    # Get nn prediction and compute loss
                    ids_pred = net(images)
                    loss = criterion(ids_pred, true_ids)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                # Update the output plotter
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                # Log the experiment values to wandb
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round at the end of every epoch
                division_step = int(n_train / batch_size) + 1
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate_recognition(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Accuracy: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation accuracy': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'ids': {
                                # Add one to the original ids, because we converted the starting index to 0
                                'true': true_ids[0].nonzero().cpu().item() + 1,
                                'pred': torch.softmax(ids_pred, dim=1).argmax(dim=1)[0].float().cpu().item() + 1,
                            },
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
    """
    Test the given neural network with the help of the training dataset
    :param net: Neural network to test
    :param device: CUDA device to use
    :param experiment: wandb experiment reference
    """
    net.eval()
    # 1. Create dataset
    test_set = BasicDatasetRecognition(dir_img_train, dict_id_translation, net.n_classes)

    # 2. Create data loader
    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    dataset_length = len(test_set)

    # Helper variable for continuous average calculation
    accuracy = 0.
    # 3. Iterate over test data and compute the average accuracy
    for batch in test_loader:
        images = batch['image']
        ids_true = batch['id']

        # Put the data onto the gpu
        images = images.to(device=device, dtype=torch.float32)
        ids_true = ids_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            # predict the ids
            id_pred = net(images)

            # Get the resulting ids from the one hot encoded vector for both
            id_pred = id_pred.argmax(dim=1).long()
            id_true = ids_true.argmax(dim=1).long()

            # Calculate accuracy as check how many ids are equivalent
            accuracy += torch.sum(torch.eq(id_pred, id_true)).item()

    accuracy = accuracy / dataset_length
    print("\n")
    print("Average accuracy:", f"{accuracy:.2%}")
    print("\n")
    if experiment is not None:
        experiment.log({
            'test accuracy': accuracy
        })


def get_args():
    parser = argparse.ArgumentParser(description='Train the CNN-model on images and target ids')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.001,
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
    net = ResNet(n_classes=100)

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
                               val_percent=args.val / 100,
                               amp=args.amp,
                               save_checkpoint=False)
        test_net(net=net,
                 device=device,
                 experiment=experiment)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
