# Classifier

This classifier is designed to train a classifier from COCO markup attributes (<i>see COCO example below</i>). The ResNet-34 network is taken as the basis for the classifier. For MultiLabel currently supports only coco attributes format. Also it has standart PyTorch ImageFolder logic for classification. 

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
- **data_type** - "IF" - for standart PyTorch ImageFolder(not multilabel) or "COCO"(supports multilabel).
- **data** - contains paths to data and folders.

    - **path_to_images** - path to the folder with images (for COCO and CSV data_type).
    - **path_to_train** - path to json with in COCO format(data_type="COCO") or path to ImageFolder(data_type="IF") or path to csv(data_type="CSV").
    - **path_to_val** - path to json with in COCO format(data_type="COCO") or path to ImageFolder(data_type="IF") or path to csv(data_type="CSV").
    - **path_to_test** - path to json with in COCO format(data_type="COCO") or path to ImageFolder(data_type="IF") or path to csv(data_type="CSV").
    - **path_to_test_result_output_folder** - path to the folder where *xlsx* files with test results will be saved.
    - **path_to_pytorch_pretrained_model** - path to pretrained [weights-RN34](https://download.pytorch.org/models/resnet34-b627a593.pth).
- **classifier** - classifier settings.
    - **resnet_layers** - number of ResNet layers. Accepted as an argument when initializing resnet. For different ResNet, a different amount is used.
    - **block_str** - "BasicBlock" or "Bottleneck" for different resnet_layers(for RN34 use BasicBlock for RN50 use Bottleneck)
    - **num_classes** - the number of classes for each argument, which are registered in *keys_outputs*.
    - **keys_outputs** - arguments that are used when training the model.
    - **categorical** - If your data (annotations) contain categorical-string values they must be spelled out in <i>categorical</i>. In <i>keys_outputs</i> you specify the key in COCO annotations, in <i>categorical</i> you specify values used in your annotations. See config exmpls for more info.

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
- **trt_precision** - fp16/fp32 (int8 to come) 
# TensorRT (TODO)

So far, only compilation and inference in TRT has been added. Test will be added soon.
You can check <i>build_trt.py</i> and <i>./scripts/TensorRT/</i> folder for the code. 

Use docker-compose to build it. Select TRT precision in configs (only fp32 and fp16 | int8 need some fixes)

Edit docker-compose.yml: add volume to weights folder at least

#### P.S. you can check <i>./scripts/Classifier.py -> class ClassifierNew</i> for TensorRT modeling with torch2trt. Only this class is implemented and not used anywere yet :)

### Run script 
```bash 
docker-compose -f docker/docker-compose.yml build
docker-compose -f docker/docker-compose.yml up

```

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
          "extra_attrib_0": <bool/int/str>,
          .
          .
          .
          "extra_attrib_N": <bool/int/str>,
        }
      } ...
    ]
  }
```
