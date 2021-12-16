# imagebasedbiometry2122
Repo for assignments for Image based Biometry in Ljubljana 2021/22

Instructions for assignment 02:
- Paste the dataset provided by the assignment files into the folder assignment02/data
- Train the model via running the file [train.py](assignment02/detectors/unet_segmentation/train.py)
  - you can specify parameters either on the command line or by modifying the default parameters. Comment in/out the desired model.
  - The model performance and the final test result will be published via wandb. See the link in the output to access the resulting graphs.
- For loading a specific model, comment in the corresponding [lines](https://github.com/matthi97/imagebasedbiometry2122/blob/7bec4d2081d8cdfe367b30d8710f1e01672381c1/assignment02/detectors/unet_segmentation/train.py#L261-L262) and execute the file.
