This project will perform a video classification task with visual features.

## Recommended Hardware

This code template is built based on [PyTorch](https://pytorch.org) and [Pyturbo](https://github.com/CMU-INF-DIVA/pyturbo) for Linux to fully utilize the computation of multiple CPU cores and GPUs.
SIFT feature, K-Means, and Bag-of-Words must run on CPUs, while CNN features and MLP classifiers can run on GPUs.
For GCP, an instance with 16 vCPU (e.g. `n1-standard-16`) and a Nvidia T4 GPU instance should be sufficient for the full pipeline.
During initial debugging, you are recommended to use a smaller instance to save money, e.g., `n1-standard-1` (only 1 vCPU) with Nvidia T4 or without GPU for the SIFT part.

## Install Dependencies

A set of dependencies is listed in [environment.yml](environment.yml). You can use `conda` to create and activate the environment easily.

```bash
# Start from within this repo
conda env create -f environment.yml -p ./env
conda activate ./env
```

## Check CUDA version

```bash
nvidia-smi
```
Check CUDA version of your device. And install the right version of pytorch(torchvision)[PyTorch](https://pytorch.org/get-started/previous-versions/)

## Dataset

You will continue using the data from [DATA](https://github.com/KevinQian97/11755-ISR-HW1#data-and-labels) for this work, which you should have downloaded.

If you don't have the data, download it from [AWS S3](https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data.zip) with the following commands:

```bash
# Start from within this repo
cd ./data
# Download and decompress data (no need if you still have it from HW1)
wget https://cmu-11775-vm.s3.amazonaws.com/spring2022/11775_s22_data.zip
unzip 11775_s22_data.zip
rm 11775_s22_data.zip
```

Eventually, the directory structure should look like this:

* this repo
  * code
  * data
    * videos (unzipped from 11775_s22_data.zip)
    * labels (unzipped from 11775_s22_data.zip)
  * env
  * ...


## SIFT Features

To extract SIFT features, use

```bash
# Extract train_val.csv
python code/run_sift.py data/labels/train_val.csv
```
```bash
# Extract test_for_students.csv
python code/run_sift.py data/labels/test_for_students.csv
```

By default, features are stored under `data/sift`.

To train K-Means with SIFT feature for 128 clusters, use

```bash
python code/train_kmeans.py data/labels/train_val.csv data/sift 128 sift_128
```

By default, model weights are stored under `data/kmeans`.

To extract Bag-of-Words representation with the trained model, use

```bash
python code/run_bow.py data/labels/train_val.csv sift_128 data/sift
```

By default, features are stored under `data/bow_<model_name>` (e.g., `data/bow_sift_128`).


## CNN Features

To extract CNN features, use

```bash
python code/run_cnn.py data/labels/train_val.csv
```
```bash
python code/run_cnn.py data/labels/test_for_students.csv
```

By default, features are stored under `data/cnn`.

## 3D CNN Features

To extract 3D CNN features, use

```bash
python code/run_cnn3d.py data/labels/train_val.csv
```
```bash
python code/run_cnn3d.py data/labels/test_for_students.csv
```

By default, features are stored under `data/cnn3d`.

## MLP Classifier

The training script automatically and deterministically split the `train_val` data into training and validation, so you do not need to worry about it.

To train MLP with SIFT Bag-of-Words, run

```bash
python code/run_mlp.py sift --feature_dir data/bow_sift_128 --num_features 128
```

To train MLP with CNN features, run

```bash
python code/run_mlp.py cnn --feature_dir data/cnn --num_features <num_feat>
```

By default, training logs and predictions are stored under `data/mlp/cnn/version_xxx/`.


To train MLP with 3d CNN features, run

```bash
python code/run_mlp.py cnn3d --feature_dir data/cnn3d --num_features <num_feat>
```

By default, training logs and predictions are stored under `data/mlp/cnn3d/version_xxx/`.


### This project was from CMU 11-775 Fall 2023 Homework 2
See [PDF Handout](docs/handout.pdf)