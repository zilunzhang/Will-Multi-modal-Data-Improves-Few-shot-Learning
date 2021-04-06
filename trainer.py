import pytorch_lightning as pl
from encoders import SentenceEncoder
from dataset import custom_dataset
from torchmeta.utils.data import BatchMetaDataLoader
from utils import *
import timm
import numpy as np
from models import *
from encoders import ResNet12, ConvNet
import ray.tune as tune
import pdb
import os
from torchmeta.datasets.helpers import *


class FSLTrainer(pl.LightningModule):

    def __init__(self, hpparams):

        super(FSLTrainer, self).__init__()

        self.hparams = hpparams
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.image_backbone = self.create_backbone(hpparams['emb_size'])
        self.model = eval(hpparams['model'])(num_way=hpparams['num_way'], num_shot=hpparams['num_shot'],
                                             num_query=hpparams['num_query'], model_configs=None)
        self.config = None
        self.sampling_policy = hpparams["data"]

    def create_backbone(self, emb_size):
        if self.hparams['backbone'].startswith('timm'):
            name = self.hparams['backbone'].split('-')[-1]
            backbone = timm.create_model(name, num_classes=emb_size, features_only=False, pretrained=False)
        else:
            name = self.hparams['backbone'].split('-')[0]
            if self.hparams['backbone'].startswith('ConvNetN'):
                repeat = self.hparams['backbone'].split('-')[1]
                backbone = eval(name)(emb_size=emb_size, repeat=int(repeat))
            else:
                backbone = eval(name)(emb_size=emb_size)
        return backbone

    def prepare_data(self):

        self.train_dataset = \
            custom_dataset(
                sampling_policy=self.sampling_policy,
                id_to_sentence=self.hparams['id_to_sentence'],
                sentence_to_id=self.hparams['sentence_to_id'],
                ways=self.hparams['num_way'],
                shots=self.hparams['num_shot'],
                test_shots=self.hparams['num_query'],
                meta_train=True,
                download=False,
                seed=self.hparams['seed'],
                transform=imagenet_transform(stage='train')
            )

        self.val_dataset = \
            custom_dataset(
                sampling_policy=self.sampling_policy,
                id_to_sentence=self.hparams['id_to_sentence'],
                sentence_to_id=self.hparams['sentence_to_id'],
                ways=self.hparams['num_way'],
                shots=self.hparams['num_shot'],
                test_shots=self.hparams['num_query'],
                meta_val=True,
                download=False,
                seed=self.hparams['seed'],
                transform=imagenet_transform(stage='val')
            )

    def train_dataloader(self):
        self.train_loader = BatchMetaDataLoader(self.train_dataset, batch_size=self.hparams['batch_size'],
                                                num_workers=self.hparams['num_cpu'], pin_memory=True)
        return self.train_loader

    def val_dataloader(self):
        self.val_dataloader = BatchMetaDataLoader(self.val_dataset, shuffle=False,
                                                  batch_size=self.hparams['batch_size'],
                                                  num_workers=self.hparams['num_cpu'], pin_memory=True)

        return self.val_dataloader

    def training_step(self, batch, batch_idx):
        support_data, support_text, support_labels = batch["train"]
        query_data, query_text, query_labels = batch["test"]

        train_loss, train_accuracy = self.forward(support_data, query_data, support_text, query_text, support_labels, query_labels, batch_idx)

        train_tensorboard_logs = {'training_loss': train_loss, 'training_accuracy': train_accuracy}
        self.trainer.logger.log_metrics(train_tensorboard_logs, step=self.trainer.global_step)
        pbar = {'train_acc': train_accuracy, 'train_loss': train_loss, 'lr': self.lr_scheduler.get_last_lr()}
        return {'loss': train_loss,
                'log': train_tensorboard_logs,
                'progress_bar': pbar
                }

    def validation_step(self, batch, batch_idx):
        if self.hpparams['model'] == "MAML":
            torch.set_grad_enabled(True)
        support_data, support_text, support_labels = batch["train"]
        query_data, query_text, query_labels = batch["test"]

        valid_loss, valid_accuracy = self.forward(support_data, query_data, support_text, query_text, support_labels, query_labels, batch_idx, False)

        valid_tensorboard_logs = {'valid_loss': valid_loss, 'valid_accuracy': valid_accuracy}
        self.trainer.logger.log_metrics(valid_tensorboard_logs, step=self.trainer.global_step)

        pbar_val = {'val_acc': valid_accuracy, 'validation_loss': valid_loss, 'lr': self.lr_scheduler.get_last_lr()}

        return {'loss': valid_loss,
                'log': valid_tensorboard_logs,
                'progress_bar': pbar_val
                }

    def validation_epoch_end(self, outs):

        ave_loss = []
        ave_acc = []

        for validation in outs:
            ave_loss.append(validation['log']['valid_loss'].detach().cpu().numpy())
            ave_acc.append(validation['log']['valid_accuracy'].detach().cpu().numpy())
        ave_loss = np.mean(np.array(ave_loss))
        ave_acc = np.mean(np.array(ave_acc))

        if ave_acc > self.best_val_acc:
            self.best_val_acc = ave_acc

        valid_tensorboard_logs = {'valid_accuracy_mean': ave_acc, 'valid_loss_mean': ave_loss}

        results = {'log': valid_tensorboard_logs}

        return results

    def test_step(self, batch, batch_idx):
        if self.hpparams['model'] == "MAML":
            torch.set_grad_enabled(True)
        support_data, support_text, support_labels = batch["train"]
        query_data, query_text, query_labels = batch["test"]

        if self.hparams["num_gpu"] != 0:
            support_data, query_data, support_labels, query_labels = support_data.to("cuda"), query_data.to(
                "cuda"), support_labels.to("cuda"), query_labels.to("cuda")

        test_loss, test_accuracy = self.forward(support_data, query_data, support_text, query_text, support_labels, query_labels, batch_idx, False)

        test_tensorboard_logs = {'test_loss': test_loss, 'test_accuracy': test_accuracy}
        self.trainer.logger.log_metrics(test_tensorboard_logs, step=self.trainer.global_step)

        pbar_test = {'test_acc': test_accuracy, 'test_loss': test_loss, 'lr': self.lr_scheduler.get_last_lr()}

        return {'loss': test_loss,
                'log': test_tensorboard_logs,
                'progress_bar': pbar_test
                }

    def test_epoch_end(self, outs):
        ave_loss = []
        ave_acc = []
        for test in outs:
            ave_loss.append(test['log']['test_loss'].detach().cpu().numpy())
            ave_acc.append(test['log']['test_accuracy'].detach().cpu().numpy())
        ave_loss = np.mean(np.array(ave_loss))
        std_acc = np.std(np.array(ave_acc))
        ave_acc = np.mean(np.array(ave_acc))

        if ave_acc > self.best_test_acc:
            self.best_test_acc = ave_acc

        print('test acc: {}'.format(ave_acc))

        test_tensorboard_logs = {'test_accuracy_mean': ave_acc, 'test_accuracy_std': std_acc,
                                 'test_loss_mean': ave_loss}

        results = {'log': test_tensorboard_logs}
        return results

    def forward(self, support_image_data, query_image_data, support_test_text, query_text_data, support_labels, query_labels, index, is_train=True):
        # support_data, query_data, support_labels, query_labels = original_support_data, original_query_data, original_support_labels, original_query_labels
        # (1, 80, 3, 84, 84)
        # input = torch.cat((support_data, query_data), 1)
        # img_vis(self.hpparams['num_way'], support_data, query_data, index)

        if self.hparams["model"] == "MAML":
            accuracy, ce_loss = self.model([support_image_data, query_image_data, None, None], support_labels, query_labels, is_train)
        else:
            # (bs, num_way * num_shot, emb_size)
            support_image_feature = backbone_two_stage_initialization(support_image_data, self.image_backbone)
            # (bs, num_way * num_query, emb_size)
            query_image_feature = backbone_two_stage_initialization(query_image_data, self.image_backbone)

            accuracy, ce_loss = self.model([support_image_feature, query_image_feature, None, None], support_labels, query_labels, is_train)
        loss = ce_loss
        return loss, accuracy

    def configure_optimizers(self):
        # set optimizer
        if self.config is not None:
            pass
        else:
            if self.hparams["model"] == "MAML":
                param_list = list(self.model.parameters())
            else:
                param_list = list(self.image_backbone.parameters()) + list(self.model.parameters())
            # default optimizer
            self.optimizer = torch.optim.Adam(
                params=param_list,
                lr=self.hparams['lr'],
                weight_decay=self.hparams['weight_decay']
            )

            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.hparams['lr_schedule_step_size'],
                gamma=self.hparams['lr_schedule_gamma']
            )
        return [self.optimizer], [self.lr_scheduler]
