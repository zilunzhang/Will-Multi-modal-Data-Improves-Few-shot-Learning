#!/usr/bin/env python3
import pytorch_lightning as pl
import argparse
import os
from trainer import FSLTrainer
from utils import mannul_seed_everything, imagenet_transform, trial_name_string
from torch.utils.data import DataLoader
import yaml
import jsonlines
import numpy as np
from dataset import custom_dataset


def inference(config):

    mannul_seed_everything(config['seed'])

    trainer = pl.Trainer(
            gpus=0,
        )

    fsl_trainer = FSLTrainer.load_from_checkpoint(config['ckpt_file'])

    fsl_trainer = fsl_trainer.to("cuda")

    on_gpu = True if config['num_gpu'] > 0 else False

    test_dataset = custom_dataset(
        config['fixed_episode_file'],
        config['dataset_root'],
        ways=config['num_way'],
        shots=config['num_shot'],
        test_shots=config['num_query'],
        meta_test=True,
        on_gpu=on_gpu,
        seed=config['seed'],
        transform=imagenet_transform(stage='test')

    )
    test_dataloader = DataLoader(test_dataset, num_workers=config["num_cpu"], pin_memory=True)
    test_result = trainer.test(model=fsl_trainer, test_dataloaders=test_dataloader)
    print("result structure")
    print(test_result)
    return test_result[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_file', type=str, default="../task_files/mini_imagenet.yaml",
                        help='path of task file')
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu per trail')
    parser.add_argument('--num_cpu', type=int, default=16,
                        help='number of cpu per trail')

    args = parser.parse_args()

    config = dict()
    with open(args.task_file) as file:
        task_yaml = yaml.load(file, Loader=yaml.FullLoader)
    num_test_episode_list = np.arange(1000, 41000, 1000)
    result_matrix = np.zeros((len(num_test_episode_list), 2))
    i = 0
    for num_test_episode in num_test_episode_list:
        sampling_file_dir = task_yaml["SAMPLING_FILE_DIR"]
        test_episodic_file = task_yaml["SAMPLING_FILE"] + ".stationary" + ".{}".format(
            task_yaml["TEST_EPISODIC_FILE_TYPE"]) + ".{}".format(num_test_episode)
        output_sampling_test_episodic_file = os.path.join("..", sampling_file_dir, test_episodic_file)
        fixed_episodes = list(jsonlines.open(output_sampling_test_episodic_file))
        config["fixed_episode_file"] = fixed_episodes
        config["dataset_root"] = None
        config["num_way"] = task_yaml["FSL_INFO"]["N"]
        config["num_shot"] = task_yaml["FSL_INFO"]["K"]
        config["num_query"] = task_yaml["FSL_INFO"]["Q"]
        config["seed"] = 0
        config["num_gpu"] = args.num_gpu
        config["num_cpu"] = args.num_cpu
        config["ckpt_file"] = os.path.join("results", "1hg3id2t", "checkpoints", "epoch=389.ckpt")
        # config["ckpt_file"] = os.path.join("results", "gu7q6z71", "checkpoints", "epoch=179.ckpt")
        test_result = inference(config)
        test_acc_mean = test_result["test_accuracy_mean"]
        test_acc_std = test_result["test_accuracy_std"]
        result_matrix[i, 0] = test_acc_mean
        result_matrix[i, 1] = test_acc_std
        i += 1
    print(result_matrix)
    np.save("result_matrix", result_matrix)
