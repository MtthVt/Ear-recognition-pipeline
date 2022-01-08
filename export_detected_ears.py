from pathlib import Path

import torch

from detectors.unet_segmentation.unet import UNet


def export_detected_ears(detect_model, dir_img: Path, dict_translation: Path, save_path: str):


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the pre-trained model
    net = UNet(n_channels=3, n_classes=2, bilinear=True)
    checkpoint = "checkpoints/revived-night-66_cp_epoch5.pth"
    net.load_state_dict(torch.load(checkpoint, map_location=device))

    # Image directory
    dir_img_test = Path('data/ears/test/')
    dict_id_translation = Path('../data/perfectly_detected_ears/annotations/recognition/awe-translation.csv')
    export_path = Path('data/unet/')
    export_detected_ears(net, dir_img_test, dict_id_translation, export_path)