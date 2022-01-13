import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_loading import BasicDatasetDetection
from metrics.evaluation import Evaluation


class EvaluateAll:

    def __init__(self):
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        with open('config.json') as config_file:
            config = json.load(config_file)

        self.images_path = config['images_path']
        self.annotations_path = config['annotations_path']

    def get_annotations(self, annot_name):
        with open(annot_name) as f:
            lines = f.readlines()
            annot = []
            for line in lines:
                l_arr = line.split(" ")[1:5]
                l_arr = [int(i) for i in l_arr]
                annot.append(l_arr)
        return annot

    def run_evaluation(self):

        im_list = sorted(glob.glob(self.images_path + '/*.png', recursive=True))
        iou_arr = []
        # preprocess = Preprocess()
        eval = Evaluation()

        # 1. Create dataset
        test_set = BasicDatasetDetection(self.images_path, self.annotations_path, 1.0)

        # 2. Create data loader
        loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, **loader_args)

        # Change the following detector and/or add your detectors below
        import detectors.cascade_detector.detector as cascade_detector
        # import detectors.your_super_detector.detector as super_detector
        cascade_detector = cascade_detector.Detector()

        # CUDA device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # UNET Segmentation network
        checkpoint = "detectors/unet_segmentation/checkpoints/checkpoint_epoch5.pth"
        import detectors.unet_segmentation.unet.unet_model as unet_model
        unet = unet_model.UNet(n_channels=3, n_classes=2, bilinear=True)
        unet.load_state_dict(torch.load(checkpoint, map_location=device))
        unet.to(device=device)

        counter = 0
        for batch in test_loader:
            images = batch['image']
            mask_true = batch['mask']
            images = images.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            # Convert to one-hot
            mask_true = F.one_hot(mask_true, unet.n_classes).permute(0, 3, 1, 2).float()

            # Get the UNET result
            mask_pred = unet.forward(images)
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), unet.n_classes).permute(0, 3, 1, 2).float()

            # Compare to true mask

            mask_pred_np = mask_pred.cpu().detach().numpy()
            mask_true_np = mask_true.cpu().detach().numpy()
            iou = eval.iou_compute(mask_pred_np[0][1], mask_true_np[0][1])
            iou_arr.append(iou)
            counter += 1
            print(counter)
            print(iou)

        miou = np.average(iou_arr)
        print("\n")
        print("Average IOU:", f"{miou:.2%}")
        print("\n")


if __name__ == '__main__':
    ev = EvaluateAll()
    ev.run_evaluation()
