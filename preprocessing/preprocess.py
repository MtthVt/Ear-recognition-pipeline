import random

import cv2
import numpy as np
import torch
import torchvision as tv
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageFilter


def histogram_equalization_rgb(img):
    """
    Equalize the histogram of the PIL image.
    """
    # Simple preprocessing using histogram equalization
    # https://en.wikipedia.org/wiki/Histogram_equalization

    im2 = ImageOps.equalize(img, mask=None)

    return im2


def image_equalization(pil_img, scale, is_mask):
    """
    Preprocesses the image.
    Convert image mode to RGB + rescale it if necessary (according to scale)
    """
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

    # Convert to RGB if image has other mode (e.g. grayscale, RGBA)
    if not is_mask and pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize image to new size if necessary
    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    return pil_img


def image_equalization_recognition(pil_img):
    """
    Preprocesses the ear image for the recognition task.
    Convert image mode to grayscale + reshape it to the predefined width
    :param pil_img: PIL image to transform
    :return: transformed PIL image
    """
    # TODO:Insert here the desired image format
    new_w, new_h = 64, 128

    # Convert to RGB if image has other mode (e.g. grayscale, RGBA)
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')

    # Resize image to new size if necessary
    pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    return pil_img


def transform_tensor(img, is_mask: bool = False):
    """
    Receives a PIL image "img" and transforms it into a tensor.
    If it is a mask, delete the channel dimension.
    """
    tensor = tv.transforms.ToTensor()(img)
    if is_mask:
        # Squeeze "channel" dimension, convert to long
        tensor = torch.squeeze(tensor, dim=0)
        tensor = tensor.type(torch.LongTensor)
    return tensor


def image_augmentation(image, mask):
    """
    Apply different Image augmentation techniques (RandomAffine, RandomHorizontal/VerticalFlip, RandomPerspective, RandomRotation)
    Apply to img & msk simultaneously.
    """
    # add dummy channel dim to mask
    mask = torch.unsqueeze(mask, 0)

    # Random crop
    i, j, h, w = tv.transforms.RandomCrop.get_params(
        image, output_size=(340, 460))
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    # Random affine
    # angle, trnsl, scale, shear = tv.transforms.RandomAffine.get_params(
    #     [0, 359], None, None, None, None)
    # image = TF.affine(image, angle, trnsl, scale, shear)
    # mask = TF.affine(mask, angle, trnsl, scale, shear)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # Squeeze the mask back
    mask = torch.squeeze(mask, dim=0)

    # Optional: Plot the pictures
    # tv.transforms.ToPILImage()(image).show()
    # mask_show = mask.type(torch.uint8)
    # tv.transforms.ToPILImage()(mask_show * 255).show()

    return image, mask


def image_edge_detection(imageObject):
    # Apply edge enhancement filter

    # edgeEnahnced = imageObject.filter(ImageFilter.EDGE_ENHANCE)
    #
    # # Apply increased edge enhancement filter
    #
    # moreEdgeEnahnced = imageObject.filter(ImageFilter.EDGE_ENHANCE_MORE)

    # Find the edges by applying the filter ImageFilter.FIND_EDGES

    imageWithEdges = imageObject.filter(ImageFilter.FIND_EDGES)

    # # Show original image - before applying edge enhancement filters
    #
    # imageObject.show()
    #
    # # Show image - after applying edge enhancement filter
    #
    # edgeEnahnced.show()
    #
    # # Show image - after applying increased edge enhancement filter
    #
    # moreEdgeEnahnced.show()
    # Delete edges at the border
    pixels = imageWithEdges.load()  # create the pixel map
    for col in range(imageWithEdges.size[0]):  # for every col:
        pixels[col, 0] = 0
        pixels[col, imageWithEdges.size[1] - 1] = 0
    for row in range(imageWithEdges.size[1]):  # For every row
        pixels[0, row] = 0
        pixels[imageWithEdges.size[0] - 1, row] = 0

    # Make image binary by applying threshold
    # img_eq = ImageOps.equalize(imageWithEdges)
    # img_auto = ImageOps.autocontrast(imageWithEdges)
    # img_bi = imageWithEdges.convert("1")

    # imageObject.show()
    # imageWithEdges.show()
    # img_bi.show()
    # img_eq.show()
    # img_auto.show()
    # hist = imageWithEdges.histogram()
    # plt.hist(hist, bins=np.arange(0,255), log=True)
    # plt.show()
    # print("Hello")
    return imageWithEdges


class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img

    # Add your own preprocessing techniques here.
