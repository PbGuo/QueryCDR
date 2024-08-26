# [ECCV 2024] QueryCDR
This is the official PyTorch implementation of the paper [QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images](https://).

## Contents
- [Overview](#overview)
- [Contribution](#contribution)
- [Requirements](#requirements_and_dependencies)
- [Dataset](#dataset)
- [Test](#test)
- [Train](#train)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Overview
![overview](fig/overview.png)

## Contribution
* We propose QueryCDR, a Query-based Controllable Distortion Rectification network for fisheye images. Extensive experiments demonstrate that our QueryCDR can deliver superior results on a variety of distortion degrees.
* We propose the Distortion-aware Learnable Query Mechanism (DLQM), which effectively introduces the latent spatial relationships to control conditions for fisheye image rectification.
* We propose two kinds of blocks for modulating features using control conditions: the Controllable Convolution Modulating Block (CCMB) and the Controllable Attention Modulating Block (CAMB). They can effectively utilize control conditions to guide the rectification process.

## Requirements
- Linux
- Python 3.8
- Pytorch 1.13

## Dataset
#### Pre-training Dataset

For pre-training the network,  you need to download the perspective dataset [Places2](http://places2.csail.mit.edu/download.html) or [Coco](https://cocodataset.org/). Then, move the downloaded images to

```
--data_prepare/pre_picture
```
run
```
python data_prepare/get_dataset_pre.py
# Specify whether the dataset is a training or test set by mode= 'train' or mode= 'test'
```
to generate your fisheye dataset. The generated fisheye images and new GT will be placed in 

```
--dataset_pre/data/train 
--dataset_pre/gt/train  
or 
--dataset_pre/data/test
--dataset_pre/gt/test
```

#### Fine-tuning Dataset

For fine-tuning the network with various distortion degrees, you need to move the images to

```
--data_prepare/fine_picture
```

run

```
python data_prepare/get_dataset_fine.py
# Specify whether the dataset is a training or test set by mode= 'train' or mode= 'test'
```

to generate your fisheye dataset. The generated fisheye images and new GT with various distortions will be placed in 

```
--dataset_fine/data/train 
--dataset_fine/gt/train  
or 
--dataset_fine/data/test
--dataset_fine/gt/test
```

## Test
2. Prepare testing dataset and modify "input_dir", "target_dir", and "weights" in `./test_RealBlur.py`
3. Run test
```
python test_ctrl.py -c configs/querycdr_pre.json
```
## Train
#### Pre-training

1. Before pre-training, make sure that the fisheye image and corresponding GT have been placed in

```
--dataset_pre/data/train
--dataset_pre/gt/train
```

2. After that, generate your pre-training image lists

```
python dataset_pre/flist.py
```

3. The updated file paths is in

```
--flist/dataset/train.flist 
--flist/dataset/train_gt.flist 
```

4. Run pre-training

```
python train_ctrl_pre.py -c configs/querycdr_pre.json
```
#### Fine-tuning

1. Before fine-tuning, make sure that the fisheye image and corresponding GT with various distortions have been placed in

```
--dataset_fine/data/train
--dataset_fine/gt/train
```

2. After that, generate your fine-tuning image lists

```
python dataset_fine/flist.py
```

3. The updated file paths is in

```
--flist/dataset/train.flist 
--flist/dataset/train_gt.flist 
```

4. Run fine-tuning

```
python train_ctrl.py -c configs/querycdr.json -l querycdr_pre/ --loadnum x --finetune
#loadnum is the number of the pre-training weight, such as 00030, 00060 etc...
```

## Citation

If you find the code and pre-trained models useful for your research, please consider citing our paper. :blush:
```
to do
```

## Acknowledgement
The code of QueryCDR is built upon [PCN](https://github.com/uof1745-cmd/PCN), and we express our gratitude to these awesome projects.
