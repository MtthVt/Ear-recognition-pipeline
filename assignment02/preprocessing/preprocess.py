import cv2
import numpy as np
from PIL import Image


def image_equalization(pil_img, scale, is_mask):
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
    img_ndarray = np.asarray(pil_img)

    # Convert image to fit to neural network dimension structure
    if not is_mask:
        img_ndarray = img_ndarray.transpose((2, 0, 1))

    # Normalize pixel values to [0, 1]
    img_ndarray = img_ndarray / 255

    return img_ndarray


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
