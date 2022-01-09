import os
from os import listdir
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageMath

from detectors.unet_segmentation.unet import UNet
from preprocessing import preprocess


def export_detected_ears(detect_model, dir_img: Path, dict_translation: Path, save_path: Path):
    """
    Uses the given nn to predict a segmentation mask, applies it onto the specified image &
    saves it with the help of dict_translation & save_path
    :param detect_model: nn used for prediction of the masks
    :param dir_img: image directory to read the input images from
    :param dict_translation: CSV containing the translations of filenames
    :param save_path: Path to save the resulting images to
    """
    detect_model.eval()
    # Load the list of images
    images = [file for file in listdir(dir_img) if not file.startswith('.')]
    # Load the csv translation file
    trans_df = pd.read_csv(dict_translation)
    last_folder_name = os.path.basename(os.path.normpath(dir_img))

    # Iterate over all the images
    for img_name in images:
        img = Image.open(Path.joinpath(dir_img, img_name))
        # Apply image equalization
        img, img_np = prepare_img_nn(img)

        # Generate mask prediction from network
        mask_pred = detect_model(img_np)
        # Convert to one hot encoded
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
        # Extract the relevant mask (ear)
        mask_pred = mask_pred[0][1].type(torch.uint8)

        # Convert the mask to a PIL image
        mask_np = mask_pred.cpu().detach().numpy()
        mask_np *= 255
        pil_msk = Image.fromarray(mask_np)
        # Optional: Plot the mask
        # pil_msk.show()

        # Generate result image by multiplying the two
        res_img = extract_mask_from_image(img, pil_msk)
        # Optional: Plot the result image
        # res_img.show()

        # ---Save the resulting image into the folder
        # Get the corresponding pd df entry
        full_img_name = last_folder_name + "/" + img_name
        rec_fn = trans_df.loc[trans_df['Detection filename'] == full_img_name, "Recognition filename"].iloc[0]
        rec_fn = rec_fn.replace("/", "_")
        res_img.save(Path.joinpath(save_path, rec_fn))


def extract_mask_from_image(img, pil_msk):
    """
    Multiplies a grey image mask with img
    :param img: 3-channel RGB PIL Image
    :param pil_msk: 1-channel Grayscale PIL Image
    :return: 3-channel RGB PIL Image, cut out
    """
    # Split the original image into three channels
    img_bands = img.split()
    res_img_bands = list()
    # Apply bitwise and for all the channels, convert back to 8-bit because bitwise & converts to 32-bit
    for i in range(len(img_bands)):
        res_img_bands.append(ImageMath.eval("convert(a & b, 'L')", a=img_bands[i], b=pil_msk))
        # img_bands[i].show()
        # res_img_bands[i].show()
    # Merge the channels back together to get resulting image
    res_img = Image.merge(mode="RGB", bands=res_img_bands)
    return res_img


def prepare_img_nn(img):
    img = preprocess.image_equalization(img, scale=1, is_mask=False)

    # Transform to np array for further techniques
    img_np = preprocess.transform_numpy(img, is_mask=False)

    # Convert to pytorch tensor
    img_np = torch.as_tensor(img_np.copy()).float().contiguous()

    # Insert dummy batch dimension
    img_np = torch.unsqueeze(img_np, dim=0)
    return img, img_np


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the pre-trained model
    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    checkpoint = 'detectors/unet_segmentation/checkpoints/UNET_ADAM_little_loss_optim_NOHISTOGRAM.pth'
    net.load_state_dict(torch.load(checkpoint, map_location=device))

    # Image directory
    dir_img_test = Path('data/ears/test/')
    dict_id_translation = Path('data/perfectly_detected_ears/annotations/recognition/awe-translation.csv')
    export_path = Path('data/unet/')
    export_detected_ears(net, dir_img_test, dict_id_translation, export_path)
