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
import pickle as pkl


class FSLTrainer(pl.LightningModule):

    def __init__(self, hpparams):

        super(FSLTrainer, self).__init__()

        self.hparams = hpparams
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.image_backbone = self.create_backbone(hpparams['emb_size'])
        self.text_backbone = SentenceEncoder(hpparams['emb_size'])
        self.model = eval(hpparams['model'])(num_way=hpparams['num_way'], num_shot=hpparams['num_shot'],
                                             num_query=hpparams['num_query'], emb_size=hpparams['emb_size'])

    def set_config(self, config):
        self.hparams = config
        print("num_worker: {}".format(self.hparams['num_cpu']))

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

        with open(os.path.join(self.hparams["dataset_root"], "data.pkl"), "rb") as f:
            self.sampling_policy = pkl.load(f)
            f.close()
        with open(os.path.join("pkl", "id_sentence_encoder.pkl"), "rb") as f:
            self.id_to_sentence = pkl.load(f)
            f.close()
        with open(os.path.join("pkl", "sentence_id_encoder.pkl"), "rb") as f:
            self.sentence_to_id = pkl.load(f)
            f.close()

        self.train_dataset = \
            custom_dataset(
                sampling_policy=self.sampling_policy,
                id_to_sentence=self.id_to_sentence,
                sentence_to_id=self.sentence_to_id,
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
                id_to_sentence=self.id_to_sentence,
                sentence_to_id=self.sentence_to_id,
                ways=self.hparams['num_way'],
                shots=self.hparams['num_shot'],
                test_shots=self.hparams['num_query'],
                meta_val=True,
                download=False,
                seed=self.hparams['seed'],
                transform=imagenet_transform(stage='val')
            )

        self.test_dataset = custom_dataset(
            sampling_policy=self.sampling_policy,
            id_to_sentence=self.id_to_sentence,
            sentence_to_id=self.sentence_to_id,
            ways=self.hparams['num_way'],
            shots=self.hparams['num_shot'],
            test_shots=self.hparams['num_query'],
            meta_test=True,
            download=False,
            seed=self.hparams['seed'],
            transform=imagenet_transform(stage='test')
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

    def test_dataloader(self):
        self.test_dataloader = BatchMetaDataLoader(self.test_dataset, shuffle=False,
                                                   batch_size=self.hparams['batch_size'],
                                                   num_workers=self.hparams['num_cpu'], pin_memory=True)

        return self.test_dataloader

    def training_step(self, batch, batch_idx):
        support_data, support_text, support_labels = batch["train"]
        query_data, query_text, query_labels = batch["test"]
        train_loss, train_accuracy = self.forward(support_data, query_data, support_text, query_text, support_labels, query_labels, batch_idx)

        train_result = pl.TrainResult(minimize=train_loss)
        train_tensorboard_logs = {'train_loss': train_loss, 'train_acc': train_accuracy, 'lr': torch.tensor(self.lr_scheduler.get_last_lr()[0])}
        train_result.log_dict(train_tensorboard_logs, prog_bar=True, logger=True, on_step=True)

        return train_result

    # def training_epoch_end(self, outs):
    #
    #     epoch_train_losses = outs["epoch_train_loss"].detach().cpu()
    #     epoch_train_accuracy = outs['epoch_train_acc'].detach().cpu()
    #     ave_loss = epoch_train_losses.mean()
    #     ave_acc = epoch_train_accuracy.mean()
    #     train_end_result = pl.TrainResult()
    #     train_end_tensorboard_logs = {'train_acc_mean': ave_acc, 'train_loss_mean': ave_loss}
    #     train_end_result.log_dict(train_end_tensorboard_logs, prog_bar=True, logger=True, on_step=True)
    #     return train_end_result

    def validation_step(self, batch, batch_idx):
        if self.hparams['model'] == "MAML":
            torch.set_grad_enabled(True)
        support_data, support_text, support_labels = batch["train"]
        query_data, query_text, query_labels = batch["test"]
        valid_loss, valid_accuracy = self.forward(support_data, query_data, support_text, query_text, support_labels, query_labels, batch_idx, False)
        val_result = pl.EvalResult()
        valid_tensorboard_logs = {'val_loss': valid_loss, 'val_acc': valid_accuracy}
        val_result.log_dict(valid_tensorboard_logs, prog_bar=True, logger=True, on_step=True)

        return val_result

    def validation_epoch_end(self, outs):
        epoch_valid_losses = outs["epoch_val_loss"].detach().cpu()
        epoch_valid_accuracy = outs['epoch_val_acc'].detach().cpu()
        ave_loss = epoch_valid_losses.mean()
        ave_acc = epoch_valid_accuracy.mean()
        valid_end_tensorboard_logs = {'val_acc_mean': ave_acc, 'val_loss_mean': ave_loss}
        val_end_result = pl.EvalResult(checkpoint_on=ave_acc)
        val_end_result.log_dict(valid_end_tensorboard_logs, prog_bar=True, logger=True, on_step=True)

        return val_end_result

    def test_step(self, batch, batch_idx):
        if self.hparams['model'] == "MAML":
            torch.set_grad_enabled(True)
        support_data, support_text, support_labels = batch["train"]
        query_data, query_text, query_labels = batch["test"]

        if self.hparams["num_gpu"] != 0:
            support_data, query_data, support_labels, query_labels = support_data.to("cuda"), query_data.to(
                "cuda"), support_labels.to("cuda"), query_labels.to("cuda")
        test_loss, test_accuracy = self.forward(support_data, query_data, support_text, query_text, support_labels, query_labels, batch_idx, False)

        test_result = pl.EvalResult()
        test_tensorboard_logs = {'test_loss': test_loss, 'test_acc': test_accuracy}
        test_result.log_dict(test_tensorboard_logs, prog_bar=True, logger=True, on_step=True)

        return test_result

    def test_epoch_end(self, outs):

        epoch_test_losses = outs["epoch_test_loss"].detach().cpu().numpy()
        epoch_test_accuracy = outs['epoch_test_acc'].detach().cpu().numpy()
        ave_loss = epoch_test_losses.mean()
        ave_acc = epoch_test_accuracy.mean()
        std_acc = epoch_test_accuracy.std()

        test_end_result = pl.EvalResult()
        test_end_tensorboard_logs = {'test_acc_mean': ave_acc, 'test_acc_std': std_acc, 'test_loss_mean': ave_loss}
        test_end_result.log_dict(test_end_tensorboard_logs, prog_bar=True, logger=True, on_step=True)
        return test_end_result

    def forward(self, support_image_data, query_image_data, support_text_data, query_text_data, support_labels, query_labels, index, is_train=True):

        # (1, 80, 3, 84, 84)
        # support_data, query_data, support_labels, query_labels = original_support_data, original_query_data, original_support_labels, original_query_labels
        # input = torch.cat((support_data, query_data), 1)
        # img_vis(self.hpparams['num_way'], support_data, query_data, index)

        # (bs, num_way * num_shot, emb_size)
        support_text_feature = backbone_sentence_embedding(support_text_data, self.text_backbone, self.id_to_sentence, self.hparams["fusion_method"])
        # (bs, num_way * num_query, emb_size)
        query_text_feature = backbone_sentence_embedding(query_text_data, self.text_backbone, self.id_to_sentence, self.hparams["fusion_method"])

        if self.hparams["model"] == "MAML":
            accuracy, loss = self.model(
                [support_image_data, query_image_data, support_text_feature, query_text_feature], support_labels,
                query_labels, self.hparams["fusion_method"], is_train)
        else:
            # (bs, num_way * num_shot, emb_size)
            support_image_feature = backbone_two_stage_initialization(support_image_data, self.image_backbone,
                                                                      self.hparams["fusion_method"])
            # (bs, num_way * num_query, emb_size)
            query_image_feature = backbone_two_stage_initialization(query_image_data, self.image_backbone,
                                                                    self.hparams["fusion_method"])
            accuracy, loss = self.model(
                [support_image_feature, query_image_feature, support_text_feature, query_text_feature], support_labels,
                query_labels, self.hparams["fusion_method"], is_train)
        return loss, accuracy

    def configure_optimizers(self):
        # set optimizer
        param_list = list(self.image_backbone.parameters()) + list(self.text_backbone.parameters()) + list(self.model.parameters())

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
