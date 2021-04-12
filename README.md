# CSC2541 Course Project By Shihao Ma, Yichun Zhang, and Zilun Zhang
## How Multimodal Data Improves Few Shot Learning

Implementation of course project of **CSC2541 Winter 2021 Topics in Machine Learning: Neural Net Training Dynamics**

Course Website : https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2021/

## Abstract


## Requirements

CUDA Version: 11.2

CUDNN Version: 8.1.1

Python : 3.8

To install dependencies:

```setup
sudo pip3 install -r requirements.txt
```
## Dataset
The main dataset is directly from links on the left, the text data and dataset split are following the paper on the middle, and the pickle version data we made could be downloaded on the right. 

|    Dataset    | Original Split + Multimodal Version Text Data | Multimodal data in PKL format|
| :-----------: |:----------------:|:----------------:|
|  [Cub_200_2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)  |  [Learning Deep Representations of Fine-grained Visual Descriptions](https://github.com/reedscot/cvpr2016)  | [Google Drive](https://drive.google.com/drive/folders/1w_SKTPg455q_2zdQjyg0rm31tikvaucL?usp=sharing)
| [vgg_102_flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | [Learning Deep Representations of Fine-grained Visual Descriptions](https://github.com/reedscot/cvpr2016) | [Google Drive]()

The dataset directory should look like this (example of cub_200_2011):
```bash
├── pkl_cub_200_2011
    ├── data.pkl
    ├── id_sentence_encoder.pkl
    ├── sentence_id_encoder.pkl
    
├── csc2541_project
    ├── main.py
    ├── trainer.py
    ├── models.py
    |── ......
```

## Training

To train the model(s) in the paper, run:

```train
python3 main.py --num_cpu 8 --num_gpu 1 --dataset_root ../pkl_cub_200_2011 --task_file config.yaml --num_epoch 100 --fusion_method fc
```


## Evaluation

To evaluate the model(s) in the paper, run:

```eval
python3 inference.py --num_cpu 8 --num_gpu 1 --test_size 600 --dataset_root ../pkl_cub_200_2011 --task_file config.yaml --ckpt_file xxx.ckpt
```

## Results
```bash
# Default checkpoints directory is:
./saves
```


[comment]: <> (## Experiment Results)

[comment]: <> (|    Backbone    | Model| Modality | Accuracy |)

[comment]: <> (| :-----------: |:----------------:|:----------------:| :----------------:|)

[comment]: <> (| [4-Conv]&#40;https://arxiv.org/abs/1605.05395&#41; | [ProtoNet]&#40;https://arxiv.org/abs/1703.05175&#41; | Image |46.99|)

[comment]: <> (| [ResNet12]&#40;https://github.com/kjunelee/MetaOptNet&#41; | [ProtoNet]&#40;https://arxiv.org/abs/1703.05175&#41; | Image |53.65|)

[comment]: <> (| [4-Conv]&#40;https://arxiv.org/abs/1605.05395&#41; | [MAML]&#40;https://arxiv.org/abs/1703.03400&#41; | Image |49.75|)

[comment]: <> (| [4-Conv]&#40;https://arxiv.org/abs/1605.05395&#41; | [ProtoNet]&#40;https://arxiv.org/abs/1703.05175&#41; | Image + Text |-|)

## Multimodal Improvement
|ID|Backbone|Model|Modality|Fusion Method|Accuracy|
|:---:|:---:|:---:|:---:|:---:|:---:|
|0|[4-Conv](https://arxiv.org/abs/1605.05395)|[ProtoNet](https://arxiv.org/abs/1703.05175)|Image|-|46.99|
|5|[4-Conv](https://arxiv.org/abs/1605.05395)|[ProtoNet](https://arxiv.org/abs/1703.05175)|Image + Text|Mean|75.52|
|6|[4-Conv](https://arxiv.org/abs/1605.05395)|[ProtoNet](https://arxiv.org/abs/1703.05175)|Image + Text|FC|73.41|
|7|[4-Conv](https://arxiv.org/abs/1605.05395)|[ProtoNet](https://arxiv.org/abs/1703.05175)|Image + Text|Attention (text guided)|78.40|
|8|[4-Conv](https://arxiv.org/abs/1605.05395)|[ProtoNet](https://arxiv.org/abs/1703.05175)|Image + Text|Attention (text residual)|63.6|
|3|[ResNet12](https://github.com/kjunelee/MetaOptNet)|[ProtoNet](https://arxiv.org/abs/1703.05175)|Image|-|53.65|
|9|[ResNet12](https://github.com/kjunelee/MetaOptNet)|[ProtoNet](https://arxiv.org/abs/1703.05175)|Image + Text|Mean|76.87|
|10|[ResNet12](https://github.com/kjunelee/MetaOptNet)|[ProtoNet](https://arxiv.org/abs/1703.05175)|Image + Text|FC|75.63|
|11|[ResNet12](https://github.com/kjunelee/MetaOptNet)|[ProtoNet](https://arxiv.org/abs/1703.05175)|Image + Text|Attention (text guided)|77.98|
|12|[ResNet12](https://github.com/kjunelee/MetaOptNet)|[ProtoNet](https://arxiv.org/abs/1703.05175)|Image + Text|Attention (text residual)|67.08|
|2|[4-Conv](https://arxiv.org/abs/1605.05395)|[MAML](https://arxiv.org/abs/1703.03400)|Image|-|49.75|
|13|[4-Conv](https://arxiv.org/abs/1605.05395)|[MAML](https://arxiv.org/abs/1703.03400)|Image + Text|Mean|51.10|
|14|[4-Conv](https://arxiv.org/abs/1605.05395)|[MAML](https://arxiv.org/abs/1703.03400)|Image + Text|FC|53.97|
|15|[4-Conv](https://arxiv.org/abs/1605.05395)|[MAML](https://arxiv.org/abs/1703.03400)|Image + Text|Attention (text guided)|Fail to Converge|
|16|[4-Conv](https://arxiv.org/abs/1605.05395)|[MAML](https://arxiv.org/abs/1703.03400)|Image + Text|Attention (text residual)|Fail to Converge|

