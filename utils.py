import torch
import os
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ColorJitter,
    CenterCrop,
)
import numpy as np
import matplotlib.pyplot as plt

# import cv2
import random
import ray
import jsonlines
import argparse
import matplotlib

matplotlib.use("Agg")


def imagenet_transform(stage):
    # mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    # std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
    # normalize = Normalize(mean=mean_pix, std=std_pix)
    # # set transformer
    # if stage == 'train':
    #     transform = Compose([
    #         RandomCrop(84, padding=4),
    #         RandomHorizontalFlip(),
    #         ColorJitter(brightness=.1,
    #                     contrast=.1,
    #                     saturation=.1,
    #                     hue=.1),
    #         lambda x: np.asarray(x),
    #         ToTensor(),
    #         normalize
    #     ]
    #     )
    # else:  # 'val' or 'test' ,
    #     transform = Compose([
    #         lambda x: np.asarray(x),
    #         ToTensor(),
    #         normalize
    #     ]
    #     )
    transform = Compose(
        [
            Resize(84),
            CenterCrop(84),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


def backbone_two_stage_initialization(full_data, encoder):
    """
    encode raw data by backbone network
    :param full_data: raw data
    :param encoder: backbone network
    :return: last layer logits from backbone network
             second last layer logits from backbone network
    """
    # encode data
    last_layer_data_temp = []
    second_last_layer_data_temp = []
    last_layer_feature_map_temp = []
    # for data in full_data.chunk(full_data.size(1), dim=1):
    #    # the encode step
    #    encoded_result = encoder(data.squeeze(1))
    #    # prepare for two stage initialization of DPGN
    #    last_layer_data_temp.append(encoded_result[0])
    #    second_last_layer_data_temp.append(encoded_result[1])
    for data in full_data:
        encoded_result = encoder(data)
        # import pdb
        # if isinstance(encoded_result, tuple):
        #    pdb.set_trace()
        last_layer_data_temp.append(encoded_result)
        second_last_layer_data_temp.append(encoded_result)
        last_layer_feature_map_temp.append(encoded_result)

    # last_layer_data: (batch_size, num_samples, embedding dimension)
    last_layer_data = torch.stack(last_layer_data_temp, dim=0)
    # second_last_layer_data: (batch_size, num_samples, embedding dimension)
    second_last_layer_data = torch.stack(second_last_layer_data_temp, dim=0)
    last_layer_feature_map = torch.stack(last_layer_feature_map_temp, dim=0)

    return last_layer_data


def backbone_sentence_embedding(full_data, encoder, id_to_sentence):
    """
    encode raw data by backbone network
    :param full_data: raw data
    :param encoder: backbone network
    :return: last layer logits from backbone network
             second last layer logits from backbone network
    """

    # encode data
    last_layer_data_temp = []

    for data in full_data:

        # (5, 800)
        encoded_result = encoder(data, id_to_sentence)

        last_layer_data_temp.append(encoded_result)

    # last_layer_data: (batch_size, num_samples, embedding dimension)
    last_layer_data = torch.stack(last_layer_data_temp, dim=0)

    # print("last_layer_data shape: {}".format(last_layer_data.shape))
    return last_layer_data


def img_vis(way_num, support_data, query_data, index, save_dir="./vis"):
    assert support_data.shape[0] == 1 and query_data.shape[0] == 1
    os.makedirs(save_dir, exist_ok=True)
    # (5, 84, 84, 3)
    support_data_permute = support_data.permute(0, 1, 3, 4, 2).squeeze(0)
    # (75, 84, 84, 3)
    query_data_permute = query_data.permute(0, 1, 3, 4, 2).squeeze(0)
    support_data_reshape = torch.reshape(
        support_data_permute, (way_num, -1, *support_data_permute.shape[1:])
    )
    query_data_reshape = torch.reshape(
        query_data_permute, (way_num, -1, *query_data_permute.shape[1:])
    )
    device = support_data.get_device()
    # device = 'cpu' if device == -1 else device
    # (5, 1+15, 84, 84, 3)
    black = torch.zeros(
        support_data_reshape.shape[0], 1, *support_data_reshape.shape[-3:]
    )
    black = black.cuda() if device != -1 else black
    complete_tensor = torch.cat(
        [support_data_reshape, black, query_data_reshape], dim=1
    )
    present_list = []
    for row in complete_tensor:
        tensor_list = [tensor for tensor in row]
        tensor_row = torch.cat(tensor_list, dim=1)
        present_list.append(tensor_row)
    present_tensor = torch.cat(present_list, dim=0)
    img = present_tensor.cpu().numpy() * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, "before_backbone_{}.png".format(index)), img)


def trans_data_order(data, label, num_way):
    """
    data: (1, 80, 128)
    label: (1, 80)
           000011111222223333344444->01234012340123401234
    """
    data_reshape = torch.reshape(data, (data.shape[0], num_way, -1, data.shape[-1]))
    data_permute = data_reshape.permute(0, 2, 1, 3)
    data_permute_flat = torch.reshape(
        data_permute, (data_permute.shape[0], -1, data_permute.shape[-1])
    )

    label_reshape = torch.reshape(label, (label.shape[0], num_way, -1))
    label_permute = label_reshape.permute(0, 2, 1)
    label_permute_flat = torch.reshape(label_permute, (label_permute.shape[0], -1))
    # return label_permute_flat
    return data_permute_flat, label_permute_flat


def get_accuracy(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum(
        (prototypes.unsqueeze(1) - embeddings.unsqueeze(2)) ** 2, dim=-1
    )
    _, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float())


# def trans_data_order(data, label, num_way):
#     """
#     data: (1, 75, 3, 84, 84)
#     label: (1, 74)
#     case:
#            000011111222223333344444 -> 01234012340123401234
#                                 or
#            01234012340123401234 -> 000011111222223333344444
#     """
#     data_reshape = torch.reshape(data, (data.shape[0], num_way, -1, *data.shape[-3:]))
#     data_permute = data_reshape.permute(0, 2, 1, 3, 4, 5)
#     data_permute_flat = torch.reshape(
#         data_permute, (data_permute.shape[0], -1, *data_permute.shape[-3:])
#     )
#     label_reshape = torch.reshape(label, (label.shape[0], num_way, -1))
#     label_permute = label_reshape.permute(0, 2, 1)
#     label_permute_flat = torch.reshape(label_permute, (label_permute.shape[0], -1))
#
#     return data_permute_flat, label_permute_flat


def mannul_seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


def trial_name_string(trial):
    """
    Args:
        trial (Trial): A generated trial object.

    Returns:
        trial_name (str): String representation of Trial.
    """
    print()
    param_dict = trial.evaluated_params
    str = ""
    # str += '{}-{}'.format('uuid', uuid)
    str += "{}-{}".format("backbone", param_dict["backbone"])
    str += "_{}-{}".format("model", param_dict["model"])
    str += "_{}-{}".format("seed", param_dict["seed"])
    str += "_{}-{}".format("lr", param_dict["lr"])
    str += "_{}-{}".format("weight_decay", param_dict["weight_decay"])
    str += "_{}-{}".format("lr_schedule_gamma", param_dict["lr_schedule_gamma"])
    str += "_{}-{}".format("lr_schedule_step_size", param_dict["lr_schedule_step_size"])
    return str
