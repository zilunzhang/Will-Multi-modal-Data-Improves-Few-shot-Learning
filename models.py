import torch.nn as nn
import torch
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from utils import get_accuracy
import torch.nn.functional as F
from torchmeta.utils.gradient_based import gradient_update_parameters
from utils import get_accuracy_maml
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)


class ProtoNet(nn.Module):
    def __init__(self, num_way, num_shot, num_query, model_configs):
        super(ProtoNet, self).__init__()
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query
        self.pred_loss = nn.CrossEntropyLoss()
        self.img_matching_loss = nn.CrossEntropyLoss()
        self.text_matching_loss = nn.CrossEntropyLoss()

        self.t = nn.Parameter(torch.Tensor(1))
        self.matching_loss_coeff = nn.Parameter(torch.Tensor(1))

    def forward(self, backbone_output, support_labels, query_labels):

        # support_image_feature: (bs, num_way * num_shot, emb_size)
        # query_image_feature: (bs, num_way * num_query, emb_size)
        # support_text_feature: (bs, num_way * num_shot, emb_size)
        # query_text_feature: (bs, num_way * num_query, emb_size)
        support_image_feature, query_image_feature, support_text_feature, query_text_feature = backbone_output

        # prototypes: (bs, num_way, emb_size)
        prototypes = get_prototypes(support_image_feature, support_labels, self.num_way)
        cls_loss = prototypical_loss(prototypes, query_image_feature, query_labels)

        loss = cls_loss

        with torch.no_grad():
            accuracy = get_accuracy(prototypes, query_image_feature, query_labels)

        return accuracy, loss


class MAML(nn.Module):
    def __init__(self, num_way, num_shot, num_query, model_configs):
        super(MAML, self).__init__()
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query
        self.pred_loss = nn.CrossEntropyLoss()
        self.img_matching_loss = nn.CrossEntropyLoss()
        self.text_matching_loss = nn.CrossEntropyLoss()
        self.feature_extractor = self.build_backbone()
        self.t = nn.Parameter(torch.Tensor(1))
        self.matching_loss_coeff = nn.Parameter(torch.Tensor(1))

    def build_backbone(self):

        def conv3x3(in_channels, out_channels, **kwargs):
            return MetaSequential(
                MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
                MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
                nn.ELU(),
                nn.MaxPool2d(2)
            )

        class ConvolutionalNeuralNetwork(MetaModule):
            def __init__(self, in_channels, out_features, hidden_size=64):
                super(ConvolutionalNeuralNetwork, self).__init__()
                self.in_channels = in_channels
                self.out_features = out_features
                self.hidden_size = hidden_size

                self.features = MetaSequential(
                    conv3x3(in_channels, hidden_size),
                    conv3x3(hidden_size, hidden_size),
                    conv3x3(hidden_size, hidden_size),
                    conv3x3(hidden_size, hidden_size)
                )

                self.classifier = MetaLinear(hidden_size * 25, out_features)

            def forward(self, inputs, params=None):
                features = self.features(inputs, params=self.get_subdict(params, 'features'))
                features = features.view((features.size(0), -1))
                logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
                return logits

        return ConvolutionalNeuralNetwork(3, 800, hidden_size=32)

    def forward(self, backbone_output, support_labels, query_labels, is_train):

        # support_image_feature: (bs, num_way * num_shot, emb_size)
        # query_image_feature: (bs, num_way * num_query, emb_size)
        # support_text_feature: (bs, num_way * num_shot, emb_size)
        # query_text_feature: (bs, num_way * num_query, emb_size)
        support_image_data, query_image_data, support_text_data, query_text_data = backbone_output
        device = torch.device('cuda' if support_image_data.is_cuda else 'cpu')
        outer_loss = torch.tensor(0., device=device)
        accuracy = torch.tensor(0., device=device)

        bs = support_image_data.shape[0]
        i = 0
        while i < bs:
            support_image_feature_i = self.feature_extractor(support_image_data[i])
            inner_loss = F.cross_entropy(support_image_feature_i, support_labels[i])
            self.feature_extractor.zero_grad()
            params = gradient_update_parameters(self.feature_extractor,
                                                inner_loss,
                                                step_size=10,
                                                first_order=False)

            query_image_feature_i = self.feature_extractor(query_image_data[i], params=params)
            outer_loss += F.cross_entropy(query_image_feature_i, query_labels[i])

            with torch.no_grad():
                accuracy += get_accuracy_maml(query_image_feature_i, query_labels[i])
            i += 1

        outer_loss.div_(bs)
        accuracy.div_(bs)
        if is_train:
            return accuracy, outer_loss
        else:
            return accuracy, outer_loss.detach().cpu()
