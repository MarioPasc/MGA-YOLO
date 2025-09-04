# Tools module

In order to compare the improved MGA-YOLO to the base Ultralytics models, we needed to make some custom modifications to the base YOLOv8. This module contains those modifications, which are intended to run using a different conda environment, with the default Ultralytics installation + pandas (`pip install ultralytics pandas`). 

You can run a training that saves the feature maps with `python -m tools.cli.ultra_train_fm --cfg tools/configs/ultra_defaults.yaml`