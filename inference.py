import pytorch_lightning as pl
import argparse
from trainer import FSLTrainer
from utils import mannul_seed_everything, imagenet_transform
from torch.utils.data import DataLoader
import yaml
from dataset import *


def inference(config):

    mannul_seed_everything(config['seed'])

    trainer = pl.Trainer(
            gpus=0,
        )

    fsl_trainer = FSLTrainer.load_from_checkpoint(config['ckpt_file'])
    print("fsl trainer loaded")
    fsl_trainer = fsl_trainer.to("cuda")

    test_dataset = custom_dataset(
        sampling_policy=config["data"],
        id_to_sentence=config['id_to_sentence'],
        sentence_to_id=config['sentence_to_id'],
        ways=config['num_way'],
        shots=config['num_shot'],
        test_shots=config['num_query'],
        meta_test=True,
        download=False,
        seed=config['seed'],
        transform=imagenet_transform(stage='test')
    )

    test_dataloader = BatchMetaDataLoader(test_dataset, shuffle=False, batch_size=config['batch_size'], num_workers=config['num_cpu'], pin_memory=True)
    test_result = trainer.test(model=fsl_trainer, test_dataloaders=test_dataloader)
    print("result structure")
    print(test_result)
    return test_result[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_file', type=str, default="config.yaml",
                        help='path of task file')
    parser.add_argument('--ckpt_file', type=str, required=True,
                                    help='path of ckpt file')
    parser.add_argument('--test_episode_num', type=str, default=2000)
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu per trail')
    parser.add_argument('--num_cpu', type=int, default=16,
                        help='number of cpu per trail')

    args = parser.parse_args()

    config = dict()
    with open(args.task_file) as file:
        task_yaml = yaml.load(file, Loader=yaml.FullLoader)

    config["dataset_root"] = None
    config["num_way"] = task_yaml["FSL_INFO"]["N"]
    config["num_shot"] = task_yaml["FSL_INFO"]["K"]
    config["num_query"] = task_yaml["FSL_INFO"]["Q"]
    config["seed"] = 0
    config["num_gpu"] = args.num_gpu
    config["num_cpu"] = args.num_cpu
    config["ckpt_file"] = args.ckpt_file
    test_result = inference(config)
    test_acc_mean = test_result["test_accuracy_mean"]
    test_acc_std = test_result["test_accuracy_std"]
    print("{} test episode, test accuracy: {1:.2g}\u00B1{0:.2g} ".format(args.test_episode_num, test_acc_mean, test_acc_std ))

