# Ear recognition pipeline (Segmentation/Detection + Recognition)
This repository contains files to establish an ear recognition pipeline.  
In the folders "detectors", one can find some approaches for ear detection (bounding boxes)
and also for the segmentation task (e.g. unet_segmentation).  
In the folder "feature_extractors", there are some approaches on the recognition task assigning
an id of a 100 class problem to an input ear.

## Instructions for assignment 02 / Detection-Segmentation:
- Paste the dataset provided by the assignment files into the folder "data"
- Train the model via running the file [train_detection.py](detectors/unet_segmentation/train_detection.py)
  - you can specify parameters either on the command line or by modifying the default parameters. Comment in/out the desired model.
  - The model performance and the final test result will be published via wandb. See the link in the output to access the resulting graphs.

## Instructions for assignment 03 / Recognition:
- Paste the dataset provided by the assignment files into the folder data.
Adjust the parameters in the file [train_recognition](feature_extractors/train_recognition.py) (line 17-23) accordingly.
- Train the model via running the file [train_recognition.py](feature_extractors/train_recognition.py)
  - you can specify parameters either on the command line or by modifying the default parameters. Comment in/out the desired model (please keep in mind that different models need different data structures. For more information see the comments in the code).
  - The model performance and the final test result will be published via wandb. See the link in the output to access the resulting graphs.
