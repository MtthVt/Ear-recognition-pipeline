import csv
import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F

from preprocessing import preprocess


class BasicDatasetDetection(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # Apply image equalization
        img = preprocess.image_equalization(img, self.scale, is_mask=False)
        mask = preprocess.image_equalization(mask, self.scale, is_mask=True)

        # Apply image preprocessing
        # img = preprocess.histogram_equalization_rgb(img)

        # Transform to np array for further techniques
        img_tensor = preprocess.transform_tensor(img)

        mask_tensor = preprocess.transform_tensor(mask, isMask=True)

        return {
            'image': img_tensor.contiguous(),
            'mask': mask_tensor.contiguous()
        }


class TransformDatasetDetection(BasicDatasetDetection):
    def __init__(self, images_dir, masks_dir, length, scale=1):
        super().__init__(images_dir, masks_dir, scale)
        # Resize the ids to meet the specified length
        self.ids = np.resize(self.ids, length)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # Apply image equalization
        img = preprocess.image_equalization(img, self.scale, is_mask=False)
        mask = preprocess.image_equalization(mask, self.scale, is_mask=True)

        # Apply image preprocessing
        # img = preprocess.histogram_equalization_rgb(img)

        # Transform to tensor for further techniques
        img_tensor = preprocess.transform_tensor(img)

        mask_tensor = preprocess.transform_tensor(mask, isMask=True)

        # Use image augmentation
        img, msk = preprocess.image_augmentation(img_tensor, mask_tensor)

        return {
            'image': img.float().contiguous(),
            'mask': msk.long().contiguous()
        }


class BasicDatasetRecognition(Dataset):
    """
    Preferred dataset to be used with the recognition task.
    """

    def __init__(self, images_dir: Path, dict_id_translation: Path, num_classes: int):
        self.images_dir = str(images_dir)
        self.dict_id_translation = dict_id_translation
        self.num_classes = num_classes

        # Save every filename with the last folder name (train/test) in list
        self.last_folder_name = os.path.basename(os.path.normpath(images_dir))
        self.img_files = [file for file in listdir(images_dir) if not file.startswith('.')]
        if not self.img_files:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        # Load the corresponding ids
        with open(dict_id_translation, mode='r') as infile:
            reader = csv.reader(infile)
            self.id_dict = {rows[0]: rows[1] for rows in reader}
        logging.info(f'Creating dataset with {len(self.img_files)} examples')

    def __len__(self):
        return len(self.img_files)

    @classmethod
    def load(cls, filename: str):
        return Image.open(filename)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        id_img = int(self.id_dict[self.last_folder_name + '/' + img_file])
        id_img -= 1  # Ids for the dataset start at 1, while our vector starts at 0
        # convert id to one hot encoded tensor
        id_img = torch.as_tensor(id_img).long()
        id_img = F.one_hot(id_img, num_classes=self.num_classes)

        # Load the image and apply preprocessing techniques
        img = self.load(self.images_dir + '/' + img_file)

        # Apply image equalization
        img = preprocess.image_equalization_recognition(img)

        # Apply image preprocessing
        # img = preprocess.histogram_equalization_rgb(img)
        img = preprocess.image_edge_detection(img)

        # Transform to np array for further techniques
        img = preprocess.transform_numpy_recognition(img)

        return {
            'image': torch.as_tensor(img.copy()).float(),
            'id': id_img.float()
        }


if __name__ == "__main__":
    # tmp = BasicDatasetRecognition("../data/perfectly_detected_ears/train",
    #                               "../data/perfectly_detected_ears/annotations/recognition/ids.csv")
    detection = TransformDatasetDetection(Path("../data/ears/train"),
                                        Path("../data/ears/annotations/segmentation/train"), 750)
    for i in range(10):
        detection.__getitem__(i)
