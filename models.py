import torch.nn as nn
import torch
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from utils import get_accuracy


class ProtoNet(nn.Module):
    def __init__(self, num_way, num_shot, num_query, model_configs):
        super(ProtoNet, self).__init__()
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query
        self.pred_loss = nn.CrossEntropyLoss()

    def forward(self, backbone_output, support_labels, query_labels):
        """
        Torchmeta Example
        """
        [last_layer_data_support, _, _], [last_layer_data_query, _, _] = backbone_output

        # support data: (task_num, num_way * num_support, dim)
        # query data: (task_num, num_way * num_query, dim)
        support_data, query_data = last_layer_data_support, last_layer_data_query

        # prototypes: (task_num, num_way, dim)
        prototypes = get_prototypes(support_data, support_labels, self.num_way)

        loss = prototypical_loss(prototypes, query_data, query_labels)

        with torch.no_grad():
            accuracy = get_accuracy(prototypes, query_data, query_labels)

        return accuracy, loss
