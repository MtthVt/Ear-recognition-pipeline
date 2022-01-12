import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchvision as tv
from tqdm import tqdm

import preprocessing.preprocess
import wandb
from evaluate import evaluate_recognition
from feature_extractors.resnet.resnet_model import ResNet
from utils.data_loading import BasicDatasetRecognition

dir_img_train = Path('../data/perfectly_detected_ears/train/')
dir_img_test = Path('../data/perfectly_detected_ears/test/')
dir_img_own_test = Path('../data/unet/segmented')
dict_id_translation = Path('../data/perfectly_detected_ears/annotations/recognition/ids.csv')
dict_id_own_translation = Path('../data/unet/ids.csv')
dict_awe_translation = Path('../data/perfectly_detected_ears/annotations/recognition/awe-translation.csv')
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
    dataset = BasicDatasetRecognition(dir_img_train, dict_id_translation)

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
                                  architecture="Resnet-own", augmentation="grayscale+edge-detection"))

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

    # Init wandb informations
    wandb_table = wandb.Table(columns=["step", "epoch", "id", "id_prediction"])
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
                true_ids = true_ids.to(device=device, dtype=torch.long)

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
                        # for tag, value in net.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate_recognition(net, val_loader, device)
                        # scheduler.step(val_score)

                        logging.info('Validation Accuracy: {}'.format(val_score))
                        for i in range(len(images)):
                            true_id = true_ids[i].cpu().item() + 1
                            pred_id = torch.softmax(ids_pred, dim=1).argmax(dim=1)[i].float().cpu().item() + 1
                            wandb_table.add_data(global_step, epoch, true_id, pred_id)
                            # wandb_table.add_data(global_step, epoch, wandb.Image(images[i]), true_id, pred_id)
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation accuracy': val_score,
                            # 'images': wandb.Image(images[0].cpu()),
                            # 'images': [wandb.Image(im) for im in images],
                            # [wandb.Image(im) for im in images_t]
                            # 'ids': {
                            #     # Add one to the original ids, because we converted the starting index to 0
                            #     'true': true_ids[0].nonzero().cpu().item() + 1,
                            #     'pred': torch.softmax(ids_pred, dim=1).argmax(dim=1)[0].float().cpu().item() + 1,
                            # },
                            'train predictions': wandb_table,
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            run_name = experiment.name
            model_path = Path.joinpath(dir_checkpoint, Path(run_name))
            Path(model_path).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(model_path / '{}_cp_epoch{}.pth'.format(run_name, epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')
    if save_checkpoint:
        # Log the final model to wandb
        wandb_model = wandb.Artifact(experiment.name, type="RESNET_Own",
                                     description="trained model for the own resnet-architecture")
        final_model_path = Path.joinpath(dir_checkpoint, "final")
        Path(final_model_path).mkdir(parents=True, exist_ok=True)
        wandb_model.add_dir(final_model_path)
        torch.save(net.state_dict(), str(final_model_path / '{}.pth'.format(experiment.name)))
        experiment.log_artifact(wandb_model)
    return experiment


def test_net(net, device, experiment, img_test_dir, id_translation_dict, wandb_name='test'):
    """
    Test the given neural network with the help of the training dataset
    :param net: Neural network to test
    :param device: CUDA device to use
    :param experiment: wandb experiment reference
    """
    logging.info("Starting neural network test.")
    net.eval()
    # 1. Create dataset
    test_set = BasicDatasetRecognition(img_test_dir, id_translation_dict)

    # 2. Create data loader
    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    dataset_length = len(test_set)

    # Init wandb table
    wandb_table = wandb.Table(columns=["image", "id", "id_prediction"])
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
            ids_pred = net(images)

            # Get the resulting ids from the one hot encoded vector for both
            ids_pred = ids_pred.argmax(dim=1).long()

            # Calculate accuracy as check how many ids are equivalent
            accuracy += torch.sum(torch.eq(ids_pred, ids_true)).item()

        for i in range(len(images)):
            wandb_table.add_data(wandb.Image(images[i]), ids_true[i], ids_pred[i])

    accuracy = accuracy / dataset_length
    print("\n")
    print("Average accuracy:", f"{accuracy:.2%}")
    print("\n")
    if experiment is not None:
        experiment.log({
            wandb_name + ' accuracy': accuracy,
            wandb_name + ' predictions': wandb_table
        })


def test_net_own_dataset(net, device, experiment):
    logging.info("Starting neural network test on own dataset.")
    net.eval()
    # Load the list of images
    images = [file for file in os.listdir(dir_img_own_test) if not file.startswith('.')]
    # Load the csv translation file
    trans_df = pd.read_csv(dict_awe_translation)

    counter = 0
    accuracy = 0
    # Iterate over all the images
    for img_name in images:
        img = Image.open(Path.joinpath(dir_img_own_test, img_name))
        # Apply image equalization
        img = preprocessing.preprocess.image_equalization_recognition(img)

        # Apply image preprocessing
        img = preprocessing.preprocess.image_edge_detection(img)

        # Convert to numpy array
        img_np = preprocessing.preprocess.transform_numpy_recognition(img)

        # Convert to pytorch tensor
        img_tensor = torch.as_tensor(img_np.copy(), dtype=torch.float32, device=device)

        # Insert dummy batch dimension
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

        # Generate id prediction from network
        id_pred = net(img_tensor)

        # Get the resulting ids from the one hot encoded vector
        id_pred = id_pred.argmax(dim=1).long().item()

        # Load the corresponding id
        img_name_search = img_name.replace("_", "/")
        id_true = trans_df.loc[trans_df['Recognition filename'] == img_name_search, "Class ID"].iloc[0]

        # Calculate accuracy as check if ids are equivalent
        accuracy += int(id_pred == id_true)

        counter += 1
        if counter % 50 == 0:
            logging.info("Finished " + str(counter) + " images.")
    accuracy = accuracy / len(images)
    print("\n")
    print("Average accuracy:", f"{accuracy:.2%}")
    print("\n")
    if experiment is not None:
        experiment.log({
            'test accuracy (own)': accuracy
        })


def get_args():
    parser = argparse.ArgumentParser(description='Train the CNN-model on images and target ids')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    # parser.add_argument('--load', '-f', type=str, default="checkpoints/final/colorful-bird-98.pth", help='Load model from a .pth file')

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
    # net = ResNet(n_classes=100)
    net = tv.models.resnet34(pretrained=True)
    net.fc = nn.Linear(in_features=512, out_features=100)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        # experiment = None
        experiment = train_net(net=net,
                               epochs=args.epochs,
                               batch_size=args.batch_size,
                               learning_rate=args.lr,
                               device=device,
                               val_percent=args.val / 100,
                               amp=args.amp,
                               save_checkpoint=True)
        # Evaluate on test data set
        test_net(net=net, device=device, experiment=experiment,
                 img_test_dir=dir_img_test, id_translation_dict=dict_id_translation)
        # Evaluate on the own segmentation dataset
        test_net(net=net, device=device, experiment=experiment,
                 img_test_dir=dir_img_own_test, id_translation_dict=dict_id_own_translation,
                 wandb_name="own_segmentation")
        # Evaluate on the own detection dataset
        test_net(net=net, device=device, experiment=experiment,
                 img_test_dir=Path('../data/unet/detection'),
                 id_translation_dict=Path('../data/unet/ids_detection.csv'),
                 wandb_name="own_segmentation")
        if experiment is not None:
            experiment.finish()
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
