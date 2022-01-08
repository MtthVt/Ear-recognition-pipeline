import cv2
import numpy as np
import torch
import torchvision as tv
from PIL import Image, ImageOps


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
        rgbimg = Image.new("RGB", pil_img.size)
        rgbimg.paste(pil_img)
        pil_img = rgbimg

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
    # TODO: Check resize options
    pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    return pil_img


def transform_numpy(pil_img, is_mask):
    """
    Transform the PIL image to numpy array. Normalize to [0,1] & convert array structure to fit NN.
    """
    img_ndarray = np.asarray(pil_img)

    # Convert image to fit to neural network dimension structure
    if not is_mask:
        img_ndarray = img_ndarray.transpose((2, 0, 1))

    # Normalize pixel values to [0, 1]
    img_ndarray = img_ndarray / 255

    return img_ndarray


def transform_numpy_recognition(pil_img):
    """
    Transform the PIL image to numpy array. Normalize to [0,1] & convert array structure to fit NN.
    """
    img_ndarray = np.asarray(pil_img)

    # Convert image to fit to neural network dimension structure
    img_ndarray = np.expand_dims(img_ndarray, 0)

    # Normalize pixel values to [0, 1]
    img_ndarray = img_ndarray / 255

    return img_ndarray


def image_augmentation(img, mask):
    """
    Apply different Image augmentation techniques (RandomAffine, RandomHorizontal/VerticalFlip, RandomPerspective, RandomRotation)
    Apply to img & msk simultaneously.
    """
    image_transformations = tv.transforms.RandomChoice(
        [tv.transforms.RandomAffine([0, 359], fillcolor=None), tv.transforms.RandomHorizontalFlip(p=0.2),
         tv.transforms.RandomPerspective(p=0.2), tv.transforms.RandomRotation(degrees=[0, 359]),
         tv.transforms.RandomVerticalFlip(p=0.2), tv.transforms.ColorJitter()])

    transform = tv.transforms.Compose([image_transformations])

    # Convert mask to 3 dimensional image (for transformation purpose)
    msk_img = Image.fromarray(mask * 255)
    # Convert mask to 3 dimensional image
    msk_img = msk_img.convert("RGB")
    msk_tensor = tv.transforms.ToTensor()(np.array(msk_img))
    img_tensor = torch.as_tensor(img.copy()).float().contiguous()
    # Stack the two tensors onto each other to get same transformations for both
    img_msk = torch.stack([img_tensor, msk_tensor])
    # Apply transformations
    img_msk = transform(img_msk)

    # Get back the original tensors
    img = img_msk[0]
    msk = img_msk[1]
    # Convert msk back to grayscale/0 dimension
    msk_img = tv.transforms.ToPILImage()(msk)
    msk_img = msk_img.convert("L")
    msk = tv.transforms.ToTensor()(np.array(msk_img))
    # Omit single dimension
    msk = torch.squeeze(msk)

    # Optional: Plot the pictures
    # tv.transforms.ToPILImage()(img).show()
    # tv.transforms.ToPILImage()(msk * 255).show()
    return img, msk


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
