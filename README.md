# Generating Defensive Plays in Basketball Games

[Paper Accepted by ACMMM 2018](https://dl.acm.org/doi/10.1145/3240508.3240670)

## Architechture
![](https://lh6.googleusercontent.com/dkqUgCKY5tZ-wL5eujfFVcSs6jDpb_ACROMPLquDNZVwbs2Q5nUaVcRUU24)

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
git clone https://github.com/chychen/bball_defensive_strategies_generation
cd script_generation
```

- Download the dataset. (You should put dataset right under the folder "{repo_path}/script_generation/data/")

google drive link: https://drive.google.com/file/d/1drGHwrCT1Fd-sVReimGD1SmGT10rxkw9/view?usp=sharing

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

### Moniter Training

- You can see all training details on through tensorboard including histogram, distribution, and several scalar metrics during training.

```bash
# (default url) -> {YOUR_IP_ADDR}:6006 i.e. 127.0.0.1:6006
tensorboard --logdir='version_1' &
```

- example (with latent weight penalty lambda=1.0)

![](https://lh6.googleusercontent.com/Z8T0VYV0o6DHqpENgGRxODXqsYTbXnvemH5ihrddfd6GVpgeJL2m1AOxc-s)

- example (without latent weight penalty lambda=0.0)

![](https://lh6.googleusercontent.com/5F0np2ynG-lIeBM9DK9vNmayAhpJGsr6XVJHGwJ6JZR5FniNr5cRcldhyJE)

## Evaluation

### Comparison

![](https://lh5.googleusercontent.com/HYH6p0a1PuOfs65nhbg5BBfX2NRRw-80d6WDdjlLxH8pIOmvIG-u-CfK3hE)

## Generative Results

- User Study Result summary
- Learn to defense pick and roll?

![](https://lh6.googleusercontent.com/yHs8-KTKGiGSL1tq9jKzJul8YpTRfX1kGWd-5lFoZ2k2E7T4a8zJTpMvxNY)

- More Results, please see [100-results-vedio](link)

## Data Preprocessing

Dataset comes from STATS SportVU NBA 2015-2016, containing players (x,y) and ball(x,y,z) position on the court during a game, tracked at 25 frames per second. privided by ![[link]](https://github.com/sealneaward/nba-movement-data) We further:

- Take all offensive sequences 
- Start of seqeunce: offensive player inbounds or brings ball past half-court
- End of sequence: offence takes a shot, either missed or made.

## Citation

```
@inproceedings{chen2018generating,
  title={Generating Defensive Plays in Basketball Games},
  author={Chen, Chieh-Yu and Lai, Wenze and Hsieh, Hsin-Ying and Zheng, Wen-Hao and Wang, Yu-Shuen and Chuang, Jung-Hong},
  booktitle={Proceedings of the 26th ACM international conference on Multimedia},
  pages={1580--1588},
  year={2018}
}
```
