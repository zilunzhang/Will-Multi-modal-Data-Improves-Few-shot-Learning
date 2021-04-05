# CSC2541 Course Project By Shihao Ma, Yichun Zhang, and Zilun Zhang
## How Multimodal Data Improved Few Shot Learning

Implementation of course project of **CSC2541 Winter 2021 Topics in Machine Learning: Neural Net Training Dynamics**

Course Website : https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2021/

## Abstract


## Requirements

CUDA Version: 10.0

CUDNN Version: 7.5.0

Python : 3.6.6

To install dependencies:

```setup
sudo pip3 install -r requirements.txt
```
## Dataset
The main dataset is directly from links on the left, the text data and dataset split are following the paper on the middle, and the pickle version data we made could be downloaded on the right. 

|    Dataset    | Original Split + Multimodal Version Text Data | Multimodal data in PKL format|
| :-----------: |:----------------:|:----------------:|
|  [Cub_200_2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)  |  [Learning Deep Representations of Fine-grained Visual Descriptions](https://github.com/reedscot/cvpr2016)  | [Google Drive](https://drive.google.com/drive/folders/1w_SKTPg455q_2zdQjyg0rm31tikvaucL?usp=sharing)


The dataset directory should look like this:
```bash
├── pkl
    ├── data.pkl
    ├── id_sentence_encoder.pkl
    ├── sentence_id_encoder.pkl

```

## Training

To train the model(s) in the paper, run:

```train
python3 main.py --num_cpu 16 --num_gpu 1 --dataset_root pkl --task_file config.yaml --batch_size 30 --num_epoch 100
```


## Evaluation

To evaluate the model(s) in the paper, run:

```eval
python3 inference.py --num_cpu 16 --num_gpu 1 --task_file config.yaml --ckpt_file xxx.ckpt
```

## Results
```bash
# Default checkpoints directory is:
./result_files
```


## Experiment Results
|    Backbone    | Model| Modality | Accuracy | Checkpoint|
| :-----------: |:----------------:|:----------------:| :----------------:| :----------------:|
| [4-Conv](https://arxiv.org/abs/1605.05395) | [ProtoNet](https://arxiv.org/abs/1703.05175) | Image |46.29| [here](https://drive.google.com/file/d/1IGb2OfuysWutgwD3KTAfrj1vEg8DV9Xh/view?usp=sharing)| 
| [ResNet12](https://github.com/kjunelee/MetaOptNet) | [ProtoNet](https://arxiv.org/abs/1703.05175) | Image |54.07| [here]()| 
| [4-Conv](https://arxiv.org/abs/1605.05395) | [ProtoNet](https://arxiv.org/abs/1703.05175) | Image + Text |-| [here]()| 



