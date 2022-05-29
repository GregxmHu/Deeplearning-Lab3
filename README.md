# deeplearning.disign.1
This repo is for the first assignment in the course: **Deep Learning Course Design**.
# Prerequisites
## Datasets 
* Download the original datasets in this [page](https://www.kaggle.com/competitions/dog-breed-identification/data)
* Generate development datasets by running the .py script.
  ```bash
    python data_split.py
  ```
    This will generate 5 dev-set for selecting checkpoint during training.
## Pretrained Weights
running this script to manually download the pretrained weights.
```bash
  cd pretrained_models
  wget https://download.pytorch.org/models/resnet18-f37072fd.pth
  wget https://download.pytorch.org/models/vgg11-8a719046.pth
```

# Training
The training commands is saved in [run.sh](./run.sh)
Running the following commands to finish training.
```bash
  bash run.sh your_model_name your_training_epoch your_learning_rate
```
Note that your_model_name should be selected from [ "resnet18" "senet18" "vgg11"], and your_learning_rate can be set to 5e-4, just for reference.

You can modify the run.sh according to your preference.
# Inference
Now you finished training and should generate predictions for test set.

Just run the following command:
```bash
   bash inference.sh your_model_name
```
The inference commands is saved in [inference.sh](./inference.sh)

# Contact
Source code of this project can be find at [src](./src), you should first carefully check them if have any questions. If questions can't be settled, contact me at ***hxm183083@gmail.com***.
