# inse6540-project

Table of Contents
=================

   * [Setup and Dependencies](#setup-and-dependencies)
   * [Run](#run)
      * [CIFAR-10](#cifar-10)
      * [CIFAR-100](#cifar-100)
      * [TinyImageNet](#tinyimagenet)

## Setup and Dependencies

1. Create and activate a conda environment with Python 3.12 as follows: 
```
conda create -n inse6450-env python=3.12
conda activate inse6450-env 
```
2. Install dependencies: 
```
pip install -r requirements.txt
``` 
   
## Run 
First, create a folder `~/data`, the datasets will be automatically downloaded to this folder upon running the code.
### CIFAR-10
For the CIFAR-10 code, run:

```
python main.py --known-class 2 --dataset cifar10 --gpu 0
```

### CIFAR-100

For the CIFAR-100 code, run:

```
python main.py --known-class 20 --dataset cifar100 --gpu 0
```

### TinyImageNet

For the TinyImageNet code, run:

```
python main.py --known-class 40 --dataset tinyimagenet --gpu 0
```