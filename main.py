import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
import argparse
import os
from pytorch_lightning.callbacks import EarlyStopping
from trainer import FSLTrainer
from rich.console import Console
from utils import *
from torchmeta.utils.data import BatchMetaDataLoader
import ray
from encoders import *
from models import *
from torchmeta.datasets.helpers import *
from ray import tune
from config_tune import set_up_config
from dataset import custom_dataset
import yaml
import uuid
from pytorch_lightning.callbacks import ModelCheckpoint
import pickle as pkl
import platform


def run(config):
    if config['num_gpu'] > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["SLURM_JOB_NAME"] = "bash"
    # os.environ["WANDB_API_KEY"]= "8fd7e687a9621b400944187435697160cbc9f0ef"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    exp_name = '{}' \
               '_dataset-{}' \
               '_backbone-{}' \
               '_num_shot-{}' \
               '_num_query-{}' \
               '_emb_size-{}' \
               '_bs-{}' \
               '_seed-{}' \
               '_lr-{}' \
               '_weight_decay-{}' \
               '_lr_schedule_step_size-{}' \
               '_lr_schedule_gamma-{}' \
        .format(
        config['model'],
        config['dataset'],
        config['backbone'],
        config['num_shot'],
        config['num_query'],
        config['emb_size'],
        config['batch_size'],
        config['seed'],
        config['lr'],
        config['weight_decay'],
        config['lr_schedule_step_size'],
        config['lr_schedule_gamma']
    )

    mannul_seed_everything(config['seed'])

    console = Console()
    console.log(exp_name)

    # save_root_dir = './{}'.format(config['exp_dir'])
    # os.makedirs(save_root_dir, exist_ok=True)
    # wandb_logger = WandbLogger(name=exp_name,
    #                            save_dir=save_root_dir,
    #                            project='FSL-MULTIMODAL-{}'.format(config['exp_dir']),
    #                            log_model=False,
    #                            offline=True
    #                            )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join("saves", exp_name, "checkpoints"),
        monitor='fix',
        save_top_k=5,
        mode='max',
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="fix",
        min_delta=0.0000,
        patience=10,
        mode='max'
    )

    if config['num_epoch'] > 10:
        check_val_every_n_epoch = config['num_epoch'] // 50
    else:
        check_val_every_n_epoch = 1

    trainer = pl.Trainer(
        default_root_dir=os.path.join("saves", exp_name),
        # early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        # fast_dev_run=True,
        deterministic=True,
        num_sanity_val_steps=0,
        max_epochs=config['num_epoch'],
        # logger=wandb_logger,
        gpus=config['num_gpu'],
        progress_bar_refresh_rate=1,
        log_save_interval=1,
        limit_train_batches=config['train_size'],
        limit_val_batches=config['validation_size'],
        limit_test_batches=config['test_size'],
        check_val_every_n_epoch=check_val_every_n_epoch,
    )

    if config["ckpt"] is not None:
        ckpt_path = os.path.join(config['project_root_path'], config['ckpt'])
        fsl_trainer = FSLTrainer.load_from_checkpoint(ckpt_path)
        fsl_trainer.set_config(config)
        print("FSL TRAINER LOADED FROM: {}".format(config['ckpt']))
    else:
        fsl_trainer = FSLTrainer(config)

    trainer.fit(fsl_trainer)

    print("best ckpt file path: {}, best ckpt accuracy: {}".format(checkpoint_callback.best_model_path, checkpoint_callback.best_model_score))

    test_result = trainer.test()

    print('trail：{}'.format(exp_name))
    print(test_result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_file', type=str, default="config.yaml",
                        help='path of task file')
    parser.add_argument('--budgets', type=int, default=1,
                        help='number of budgets')
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='number of gpu per trail')
    parser.add_argument('--num_cpu', type=int, default=8,
                        help='number of cpu per trail')
    parser.add_argument('--exp_dir', type=str,
                        default='result_files',
                        help='experiment name')
    parser.add_argument('--dataset_root', type=str,
                        default='../pkl_dataset',
                        help='dataset root')
    parser.add_argument('--train_size', type=int, default=100,
                        help='number of batch for train')
    parser.add_argument('--validation_size', type=int, default=100,
                        help='number of batch for validation')
    parser.add_argument('--test_size', type=int, default=600,
                        help='number of batch for test')
    parser.add_argument('--num_epoch', type=int, default=100,
                        help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of episode per batch')
    parser.add_argument('--select_func', type=str, default='grid',
                        help='function for selecting hp')
    parser.add_argument('--fusion_method', type=str, default='mean',
                        help='fusion method to text and image data')
    parser.add_argument('--ckpt', type=str,
#                         default='multimodal_best_weights/_ckpt_epoch_489.ckpt',
                        default=None,
                        help='function for selecting hp')
    args = parser.parse_args()
    print('budget, cpu, gpu: {}, {}, {}'.format(args.budgets, args.num_cpu, args.num_gpu))
    ray.init(local_mode=True)
    ray_resources = ray.available_resources()
    print('available devices: {}'.format(ray_resources))

    if args.select_func == 'grid':
        config = set_up_config(tune.grid_search)
    elif args.select_func == 'choice':
        config = set_up_config(tune.choice)
    else:
        print('invalid search function')
        exit()
    new_config = dict()
    for key in config:
        new_config[key] = config[key]["grid_search"][0]

    config = new_config
    config['task_file'] = args.task_file
    config['num_gpu'] = args.num_gpu
    config['num_cpu'] = args.num_cpu
    config['budgets'] = args.budgets
    config['dataset_root'] = args.dataset_root
    config['train_size'] = args.train_size
    config['validation_size'] = args.validation_size
    config['test_size'] = args.test_size
    config['num_epoch'] = args.num_epoch
    config['batch_size'] = args.batch_size
    config["fusion_method"] = args.fusion_method
    with open(args.task_file) as file:
        task_yaml = yaml.load(file, Loader=yaml.FullLoader)
    config['exp_dir'] = task_yaml["RESULT_FILE_DIR"]
    config['dataset'] = task_yaml["NAME"]
    config['num_way'] = task_yaml["FSL_INFO"]["N"]
    config['num_shot'] = task_yaml["FSL_INFO"]["K"]
    config['num_query'] = task_yaml["FSL_INFO"]["Q"]
    config['ckpt'] = args.ckpt
    config['project_root_path'] = os.getcwd()

    run(config)


if __name__ == '__main__':
    main()
