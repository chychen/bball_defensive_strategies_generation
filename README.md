# Basketbal Defensive Strategy Generation

[[Arxiv]]()

## Architechture
![](https://lh4.googleusercontent.com/rW_bzu4dIrRyARX2QIdMtORnf-H_G65UkBFYKh_4TbTAspRHnfu0ruy4B3E)

Tensorflow implementation for generating basketball realistic defensive strategies.
For example:

- Red Circle: Offensive Player
- Blue Circle: Defensive Player
- Green Circle: Ball

![](https://lh6.googleusercontent.com/yHs8-KTKGiGSL1tq9jKzJul8YpTRfX1kGWd-5lFoZ2k2E7T4a8zJTpMvxNY)

## Setup

### Prerequisites

- Linux
- NVIDIA GPU + CUDA CuDNN
- Tensorflow 1.4.0
- Python 3.5

### Getting Started

- Clone this repo:

```bash
git clone git@http://CGVLab:30000/nba/script_generation.git
cd script_generation
```

- Download the dataset. (You should put dataset right under the folder "{repo_path}/script_generation/data/")

```bash
cd data
wget http://140.113.210.14:6006/NBA/data/FEATURES-4.npy
```

## Training

### FLAGS

- comment (required): string, anything you want to comment for training. e.g. "first training"
- folder_path (required): string, the root folder to collect training log, checkpoints, and results. e.g. "version_1"
- others (optional): please see descriptions in [train.py](http://CGVLab:30000/nba/script_generation/src/C_WGAN_ResBlock/train.py).

### Train Model

- It cost about 2 days on single 1080 Ti to reach 500k iterations.
- By default, it save one checkpoints and one generative result for every 100 epoches, under the folders "{folder_path}/checkpoints/" and "{folder_path}/sample/"
- All hyper-perameters will be saved into json file as "{folder_path}/hyper_perameters.json"

```bash
cd ../src/C_WGAN_ResBlock/
python train.py --comment='first training' --folder_path='version_1'
```

## Evaluation

### Comparison
![](https://lh5.googleusercontent.com/HYH6p0a1PuOfs65nhbg5BBfX2NRRw-80d6WDdjlLxH8pIOmvIG-u-CfK3hE)

## Generative Results

- User Study Link?
- Learn to defense pick and roll?
![](https://lh6.googleusercontent.com/yHs8-KTKGiGSL1tq9jKzJul8YpTRfX1kGWd-5lFoZ2k2E7T4a8zJTpMvxNY)
- More Results, please see [100-results-vedio](link)

## Data Preprocessing
Dataset comes from STATS SportVU NBA 2015-2016, containing players (x,y) and ball(x,y,z) position on the court during a game, tracked at 25 frames per second. privided by ![](https://github.com/sealneaward/nba-movement-data) We further:
- Take all offensive sequences 
- Start of seqeunce: offensive player inbounds or brings ball past half-court
- End of sequence: offence takes a shot, either missed or made. 
- Jimmy
- Nsknsl

## Citation

- After Submission