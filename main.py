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
    early_stop_callback = EarlyStopping(
        monitor='valid_accuracy_mean',
        min_delta=0.00000,
        patience=50,
        verbose=True,
        mode='max'
    )

    # save_root_dir = './{}'.format(config['exp_dir'])
    # os.makedirs(save_root_dir, exist_ok=True)
    # wandb_logger = WandbLogger(name=exp_name,
    #                            save_dir=save_root_dir,
    #                            project='FSL-MULTIMODAL-{}'.format(config['exp_dir']),
    #                            log_model=False,
    #                            offline=True
    #                            )

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=os.path.join(config['exp_dir'], "checkpoints"),
    #     filename=os.path.join(exp_name, '-{epoch:02d}-{val_acc:.2f}'),
    #     monitor='val_acc',
    #     save_top_k=1,
    #     mode='max',
    # )

    if config['num_epoch'] > 10:
        check_val_every_n_epoch = config['num_epoch'] // 10
    else:
        check_val_every_n_epoch = 1

    trainer = pl.Trainer(
        # early_stop_callback=early_stop_callback,
        # callbacks=[checkpoint_callback],
        # fast_dev_run=True,
        deterministic=True,
        num_sanity_val_steps=0,
        max_epochs=config['num_epoch'],
        # logger=wandb_logger,
        gpus=config['num_gpu'],
        progress_bar_refresh_rate=1,
        # log_save_interval=1,
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
    test_result = trainer.test()

    # if type(test_result) == list:
    #     test_result = test_result[0]["test_accuracy_mean"]
    # elif type(test_result) == dict:
    #     test_result = test_result["test_accuracy_mean"]

    print('trail：{}'.format(exp_name))
    print(test_result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_file', type=str, default="config.yaml",
                        help='path of task file')
    parser.add_argument('--budgets', type=int, default=1,
                        help='number of budgets')
    parser.add_argument('--num_gpu', type=int, default=0,
                        help='number of gpu per trail')
    parser.add_argument('--num_cpu', type=int, default=0,
                        help='number of cpu per trail')
    parser.add_argument('--exp_dir', type=str,
                        # required=True,
                        default='result_files',
                        help='experiment name')
    parser.add_argument('--dataset_root', type=str,
                        default='pkl',
                        help='dataset root')
    parser.add_argument('--train_size', type=int, default=10,
                        help='number of batch for train')
    parser.add_argument('--validation_size', type=int, default=10,
                        help='number of batch for validation')
    parser.add_argument('--test_size', type=int, default=10,
                        help='number of batch for test')
    parser.add_argument('--num_epoch', type=int, default=5,
                        help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='number of episode per batch')
    parser.add_argument('--select_func', type=str, default='grid',
                        help='function for selecting hp')
    parser.add_argument('--fusion_method', type=str, default='mean',
                        help='fusion method to text and image data')
    parser.add_argument('--ckpt', type=str,
                        # default='result_files_additive_50epoch/FSL-MULTIMODAL/checkpoints/epoch_49.ckpt',
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
    # config['exp_dir'] = args.exp_dir
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

    # with torch.autograd.profiler.profile(
    #         enabled=True,
    #         # with_flops=True,
    #         use_cuda=True,
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True,
    #         use_cpu=True,
    # ) as prof:
    #     # try:
    #     #     analysis = tune.run(
    #     #         run_or_experiment=run,
    #     #         config=config,
    #     #         resources_per_trial={"cpu": config['num_cpu'], "gpu": config['num_gpu']},
    #     #         num_samples=config['budgets'] if args.select_func == 'choice' else 1,
    #     #         local_dir='{}'.format("saves"),
    #     #         trial_dirname_creator=tune.function(trial_name_string),
    #     #         queue_trials=True,
    #     #         reuse_actors=True,
    #     #     )
    #     run(config)
    # prof.export_chrome_trace('./profile.json')

    run(config)

    # os.makedirs(os.path.join("..", task_yaml["RUN_FILE_DIR"]), exist_ok=True)
    # yaml_export = os.path.join("..", task_yaml["RUN_FILE_DIR"], task_yaml["NAME"] + ".run.yaml")
    # print(yaml_export)
    # with open(yaml_export, 'w') as file:
    #     yaml.dump({"task_id": uuid.uuid4().int}, file)
    #     yaml.dump({"task_config": task_yaml}, file)
    #     yaml.dump({"model_config": config}, file)


if __name__ == '__main__':
    main()
