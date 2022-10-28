# Classifier

This classifier is designed to train a classifier from COCO markup attributes (<i>see COCO example below</i>). The ResNet-34 network is taken as the basis for the classifier.

## Start training

Install dependencies before launch

```bash
pip install -r requirements.txt
```

To start training run:

```bash
python train.py --config configs/your_config.json
```
where *your_config.json* is the config with your settings.

## Run test

To run the test run:

```bash
python test.py --config configs/your_config.json
```
where *your_config.json* is the config with your settings.

## Config

All settings for training and testing are exclusively in the configs. <br><b> You may need to tweak the Dataloader for your dataset </b>

- **config_name** - config name. This name is used when saving weights and tests.
- **data** - contains paths to data and folders.

    - **path_to_images** - path to the folder with images.
    - **path_to_train_json** - path to json with training data in COCO format.
    - **path_to_val_json** - path to json with validation data in COCO format.
    - **path_to_test_json** - path to json with test data in COCO format.
    - **path_to_test_result_output_folder** - path to the folder where *xlsx* files with test results will be saved.
    - **path_to_pytorch_pretrained_model** - path to pretrained [weights](https://download.pytorch.org/models/resnet34-b627a593.pth).
- **classifier** - classifier settings.
    - **resnet_layers** - number of ResNet layers. Accepted as an argument when initializing resnet. For different ResNet, a different amount is used.
    - **num_classes** - the number of classes for each argument, which are registered in *keys_outputs*.
    - **keys_outputs** - arguments that are used when training the model.
    - **general_types** - names of all basic types that are used in training. Only needed if *general_types* is in *keys_outputs*.
    - **types** - names of all types that are used in training. Only needed if *types* is in *keys_outputs*.
- **train_config** - training settings.
    - **weights_path** - path to the folder where the weights will be saved.
    - **start_epoch** - start epoch. The default is 0.
    - **epochs** - number of learning epochs.
    - **batch_size** - batch size.
    - **workers** - number of workers.
    - **learning_rate** - learning_rate for SGD.
    - **momentum** - momentum rate for SGD.
    - **weight_decay** - weight_decay rate for SGD.
    - **print_freq** - frequency of displaying training/validation/test information.
    - **resume** - the path to the scales you want to continue learning. If you do not want to continue training, then use null.
    - **use_criterion_weights** - use per-class weight scaling or not.
    - **use_pytorch_pretrained_model** - use pretrained model from pytorch or not.
    - **resize_h** - image height.
    - **resize_w** - image width.
    - **transform_train** - image augmentation during training.
        - **use_motion_blur** - use [motion blur](https://kornia.readthedocs.io/en/latest/augmentation.module.html#kornia.augmentation.RandomMotionBlur) during training or not.
        - **use_planckian_jitter** - use [planckian jitter](https://kornia.readthedocs.io/en/latest/augmentation.module.html#kornia.augmentation.RandomPlanckianJitter) during training or not.
        - **use_random_affine** - use [random affine](https://kornia.readthedocs.io/en/latest/augmentation.module.html#kornia.augmentation.RandomAffine) during training or not.


# COCO Format example

```json
{
    "licenses": [],
    "info": {},
    "categories": [],
    "images": [
      {
        "id": 1,
        "width": 1,
        "height": 1,
        "file_name": "1.jpg",
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0
      } ...
    ],
    "annotations": [
      {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "segmentation": [[<polygon>]],
        "area": 1,
        "bbox": [<x, y, width, height>],
        "iscrowd": 0,
        "attributes": {
          "extra_attrib_0": <bool/str>,
          .
          .
          .
          "extra_attrib_N": <bool/str>,
        }
      } ...
    ]
  }
```


