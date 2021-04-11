import torch.nn as nn
import torch
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from utils import get_accuracy
import torch.nn.functional as F
import math
import numpy as np
from torchmeta.utils.gradient_based import gradient_update_parameters
from utils import get_accuracy_maml
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)


class ProtoNet(nn.Module):
    def __init__(self, num_way, num_shot, num_query, emb_size):
        super(ProtoNet, self).__init__()
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query
        self.pred_loss = nn.CrossEntropyLoss()
        self.img_matching_loss = nn.CrossEntropyLoss()
        self.text_matching_loss = nn.CrossEntropyLoss()

        # self.t = nn.Parameter(torch.Tensor(1))
        self.t = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        self.matching_loss_coeff = nn.Parameter(torch.Tensor(1))
        self.fusion_fc = nn.Linear(2 * emb_size, emb_size)
        self.attention = Attention(emb_size, emb_size, emb_size, emb_size)

    def emb_fusion(self, support_image_feature, support_text_feature, query_image_feature, query_text_feature, mode):
        if mode == "mean":
            support_feature = (support_image_feature + support_text_feature) / 2
            query_feature = (query_image_feature + query_text_feature) / 2
        elif mode == "fc":
            # (bs, 5, 1600)
            support_feature = torch.cat((support_image_feature, support_text_feature), 2)
            # (bs, 5, 800)
            support_feature = self.fusion_fc(support_feature)
            # (bs, 75, 1600）
            query_feature = torch.cat((query_image_feature, query_text_feature), 2)
            # bs * 75 * 800
            query_feature = self.fusion_fc(query_feature)
        elif mode == "attention":
            support_feature = self.attention(support_image_feature, support_text_feature, support_text_feature)
            query_feature = self.attention(query_image_feature, query_text_feature, query_text_feature)
            # support_feature = self.attention(support_text_feature, support_image_feature, support_image_feature)
            # query_feature = self.attention(query_text_feature, query_image_feature, query_image_feature)
        else:
            "please specify a fusion method"
            exit()
        return support_feature, query_feature

    def forward(self, backbone_output, support_labels, query_labels, fusion_method, is_train):

        # support_image_feature: (bs, num_way * num_shot, emb_size)
        # query_image_feature: (bs, num_way * num_query, emb_size)
        # support_text_feature: (bs, num_way * num_shot, emb_size)
        # query_text_feature: (bs, num_way * num_query, emb_size)

        support_image_feature_pack, query_image_feature_pack, support_text_feature_pack, query_text_feature_pack = backbone_output

        if fusion_method != "attention":

            support_image_feature, query_image_feature, support_text_feature, query_text_feature = \
                support_image_feature_pack, query_image_feature_pack, support_text_feature_pack, query_text_feature_pack

            device = torch.device('cuda' if support_image_feature.is_cuda else 'cpu')
            self.t = self.t.to(device)
            self.matching_loss_coeff = self.matching_loss_coeff.to(device)
            self.fusion_fc = self.fusion_fc.to(device)

            support_feature, query_feature = self.emb_fusion(
                support_image_feature,
                support_text_feature,
                query_image_feature,
                query_text_feature,
                mode=fusion_method
            )

        else:
            support_image_feature, support_image_feature_att = support_image_feature_pack
            query_image_feature, query_image_feature_att = query_image_feature_pack
            support_text_feature, support_text_feature_att = support_text_feature_pack
            query_text_feature, query_text_feature_att = query_text_feature_pack

            device = torch.device('cuda' if support_image_feature.is_cuda else 'cpu')
            self.t = self.t.to(device)
            self.matching_loss_coeff = self.matching_loss_coeff.to(device)
            self.fusion_fc = self.fusion_fc.to(device)

            support_feature, query_feature = self.emb_fusion(
                support_image_feature_att,
                support_text_feature_att,
                query_image_feature_att,
                query_text_feature_att,
                mode=fusion_method
            )

        # prototypes: (bs, num_way, emb_size)
        prototypes = get_prototypes(support_feature, support_labels, self.num_way)
        cls_loss = prototypical_loss(prototypes, query_feature, query_labels)

        # (bs, num_way * (num_shot + num_query), emb_size)
        all_image_feature = torch.cat([support_image_feature, query_image_feature], dim=1)
        # (bs, num_way * (num_shot + num_query), emb_size)
        all_text_feature = torch.cat([support_text_feature, query_text_feature], dim=1)
        # # (bs, num_way * (num_shot + num_query), )
        # all_label = torch.cat([support_labels, query_labels], dim=1)

        # l2 norm for clip loss
        all_text_feature = F.normalize(all_text_feature, dim=-1)
        all_image_feature = F.normalize(all_image_feature, dim=-1)

        def clip_loss(all_image_feature, all_text_feature):
            #  clip the temperature from 0 to log100
            self.t.data = torch.clamp(self.t.data, 0, 4.605170185988092)
            # torch.clamp(self.model.module.logit_scale.data, 0, 4.6052)
            assert all_image_feature.shape[0] == all_text_feature.shape[0]
            bs = all_image_feature.shape[0]
            i = 0
            total_loss = 0
            while i < bs:
                # (80, 128)
                img_feat = all_image_feature[i]
                # (80, 128)
                text_feat = all_text_feature[i]
                scale = torch.exp(self.t)
                logits_per_image = scale * img_feat @ text_feat.t()
                logits_per_text = scale * text_feat @ img_feat.t()
                labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
                image_loss = F.cross_entropy(logits_per_image, labels)
                text_loss = F.cross_entropy(logits_per_text, labels)
                contrastive_loss = (image_loss + text_loss) / 2
                total_loss += contrastive_loss
                i += 1
            return total_loss / bs

        matching_loss = clip_loss(all_image_feature, all_text_feature)
        # loss = cls_loss + self.matching_loss_coeff * matching_loss
        loss = cls_loss + matching_loss
        with torch.no_grad():
            accuracy = get_accuracy(prototypes, query_feature, query_labels)
        return accuracy, loss


class Attention(torch.nn.Module):

    def __init__(self, q_dim, k_dim, v_dim, h_dim):
        super().__init__()
        self.q = nn.Linear(q_dim, h_dim)
        self.k = nn.Linear(k_dim, h_dim)
        self.v = nn.Linear(v_dim, h_dim)
        # a two-layer MLP: FC(d)-ReLU-Dropout(0.1)-FC(1)
        self.mlp = nn.Sequential(
            nn.Linear(10, 5),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(5, 1),
        )
        self.reduce_att = nn.Sequential(
            nn.Linear(v_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def similarity(self, attention_type, query, key, choice="mixed"):
        """
        Similarity Function of Attention
        :param attention_type: additive, scale dot, mlp
        :param query: (bs, 5, 10, 128), image
        :param key: (bs, 5, 10, 128), text
        :param choice: instance (along 1-axis), bs (along 0-axis), mixed (flatten all channel then along it (5, 10) -> (50, ))
        :return: (bs, 5, 10, 10) (bs or instance) or (bs, 50, 50) (mixed)
        """

        device = torch.device('cuda' if query.is_cuda else 'cpu')

        # shape problem
        if choice == "instance":
            # loop through instance axis
            num_instance = query.shape[1]
            # (bs, 5, 10, 10)
            result_matrix = torch.zeros(query.shape[0], query.shape[1], query.shape[2], query.shape[2]).to(device)
            i = 0
            while i < num_instance:
                # (bs, 10, 10)
                scores = torch.bmm(query[:, i, :, :], key[:, i, :, :].permute(0, 2, 1)) / math.sqrt(key[:, i, :, :].shape[-1])
                result_matrix[:, i, :, :] = scores
                i += 1

        # shape problem
        elif choice == "bs":
            # loop through bs axis
            num_bs = query.shape[0]
            # (bs, 5, 10, 10)
            result_matrix = torch.zeros(query.shape[0], query.shape[1], query.shape[2], query.shape[2]).to(device)
            i = 0
            while i < num_bs:
                # (5, 10, 128) x (5, 128, 10) -> (5, 10, 10)
                # For each instance, we have attention map between two modalities, total 5 instances.
                scores = torch.bmm(query[i], key[i].permute(0, 2, 1)) / math.sqrt(key[i].shape[-1])
                # each batch has batch size number of episodes
                result_matrix[i] = scores
                i += 1

        else:
            assert key.shape == query.shape
            # flatten all
            # (bs, 5 * 10, 128)
            key = torch.reshape(key, (key.shape[0], key.shape[1] * key.shape[2], key.shape[3]))
            # (bs, 5 * 10, 128)
            query = torch.reshape(query, (query.shape[0], query.shape[1] * query.shape[2], query.shape[3]))
            # (bs, 5 * 10, 5 * 10)
            result_matrix = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(key.shape[-1])

        return result_matrix

    def forward(self, query, key, value):
        """
        Attention is all you need: https://arxiv.org/pdf/1706.03762.pdf
        :param query: (bs, 5, 10, 128), image
        :param key: (bs, 5, 10, 128), text
        :param value: (bs, 5, 10, 128), text
        :return: (bs, 5, 10, 128)
        """
        device = torch.device('cuda' if query.is_cuda else 'cpu')
        self.mlp = self.mlp.to(device)
        self.reduce_att = self.reduce_att.to(device)
        self.q = self.q.to(device)
        self.k = self.k.to(device)
        self.v = self.v.to(device)

        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        # # The mixed option
        # # (bs, 50, 50)
        # scores = self.similarity("scaled_dot_product", query, key, choice="mixed")
        # att_map = F.softmax(scores, dim=-1)
        # # att_map = scores / torch.sum(scores, dim=-1, keepdim=True)
        # value = torch.reshape(value, (value.shape[0], value.shape[1] * value.shape[2], value.shape[3]))
        # # (bs, 50, 50) x (bs, 50, 128) -> (bs, 50, 128)
        # context = torch.bmm(att_map, value)
        # # (bs, 5, 10, 128)
        # context = torch.reshape(context, key.shape)
        # # (bs, 5, 128, 10)
        # permute_context = context.permute(0, 1, 3, 2)
        # # (bs, 5, 128, 1)
        # context = self.mlp(permute_context).squeeze(-1)

        # The bs option
        # (bs, 5, 10, 10)
        # each row: probability of an image channel with corresponding 10 sentence, sum to 1
        scores = self.similarity("scaled_dot_product", query, key, choice="bs")
        # softmax [(Q x V) / sqrt (k)]
        att_map = F.softmax(scores, dim=-1)

        # loop through bs axis
        num_bs = value.shape[0]
        # (bs, 5, 10, 128)
        context = torch.zeros(value.shape[0], value.shape[1], value.shape[2], value.shape[3]).to(device)
        i = 0
        while i < num_bs:
            # apply attention map
            # each row: each image channel's corresponding text embedding (weighted by attention map)
            # (5, 10, 10) x (5, 10, 128) -> (5, 10, 128)
            context[i] = torch.bmm(att_map[i], value[i])
            i += 1

        # 128 dim reduce to 1 dim through mlp
        # (bs, 5, 10, 1)
        att_reduce_weight = self.reduce_att(context)
        # (bs, 5, 128, 1)
        attended_content = torch.zeros(context.shape[0], context.shape[1], context.shape[3], att_reduce_weight.shape[3]).to(device)
        i = 0
        while i < num_bs:
            # weighted average
            # (5, 128, 10) x (5, 10, 1) -> (5, 128, 1)
            attended_content[i] = torch.bmm(context[i].permute(0, 2, 1), att_reduce_weight[i])
            i += 1
    
        return attended_content.squeeze(-1)


class MAML(nn.Module):
    def __init__(self, num_way, num_shot, num_query, emb_size):
        super(MAML, self).__init__()
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query
        self.pred_loss = nn.CrossEntropyLoss()
        self.img_matching_loss = nn.CrossEntropyLoss()
        self.text_matching_loss = nn.CrossEntropyLoss()
        self.feature_extractor = self.build_backbone(emb_size)
        self.t = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        self.matching_loss_coeff = nn.Parameter(torch.tensor(1.))
        self.fusion_fc = nn.Linear(2 * emb_size, emb_size)
        self.attention = Attention(emb_size, emb_size, emb_size, emb_size)

    def build_backbone(self, emb_size):

        def conv3x3(in_channels, out_channels, **kwargs):
            return MetaSequential(
                MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
                MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
                nn.ELU(),
                nn.MaxPool2d(2)
            )

        class ConvolutionalNeuralNetwork(MetaModule):
            def __init__(self, in_channels, out_features, hidden_size=32):
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
                self.image_conv1x1 = nn.Conv1d(1, 10, 1)

            def forward(self, inputs, fusion_method, params=None):
                features = self.features(inputs, params=self.get_subdict(params, 'features'))
                features = features.view((features.size(0), -1))
                logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))

                if fusion_method == "attention":
                    att_flatten_x = self.image_conv1x1(features.unsqueeze(1))
                    out_att = self.classifier(att_flatten_x)
                    return logits, out_att
                else:
                    return logits
        return ConvolutionalNeuralNetwork(3, emb_size, hidden_size=32)

    def emb_fusion(self, image_feature, text_feature, mode):
        if mode == "mean":
            feature = (image_feature + text_feature) / 2
        elif mode == "fc":
            # (bs, 5, 1600)
            feature = torch.cat((image_feature, text_feature), 2)
            # (bs, 5, 800)
            feature = self.fusion_fc(feature)
        elif mode == "attention":
            feature = self.attention(image_feature, text_feature, text_feature)
            # feature = self.attention(text_feature, image_feature, image_feature)
        else:
            "please specify a fusion method"
            exit()
        return feature

    def forward(self, backbone_output, support_labels, query_labels, fusion_method, is_train):

        # support_image_feature: (bs, num_way * num_shot, emb_size)
        # query_image_feature: (bs, num_way * num_query, emb_size)
        # support_text_feature: (bs, num_way * num_shot, emb_size)
        # query_text_feature: (bs, num_way * num_query, emb_size)

        support_image_feature_pack, query_image_feature_pack, support_text_feature_pack, query_text_feature_pack = backbone_output

        if fusion_method != "attention":
            support_image_data, query_image_data, support_text_feature, query_text_feature = \
                support_image_feature_pack, query_image_feature_pack, support_text_feature_pack, query_text_feature_pack

            device = torch.device('cuda' if support_image_data.is_cuda else 'cpu')
            self.t = self.t.to(device)
            self.matching_loss_coeff = self.matching_loss_coeff.to(device)
            self.fusion_fc = self.fusion_fc.to(device)

            support_image_feature = torch.zeros(support_text_feature.shape).to(device)
            query_image_feature = torch.zeros(query_text_feature.shape).to(device)

            outer_loss = torch.tensor(0., device=device)
            accuracy = torch.tensor(0., device=device)
            bs = support_image_data.shape[0]
            i = 0
            while i < bs:
                support_image_feature_i = self.feature_extractor(support_image_data[i], fusion_method)
                support_text_feature_i = support_text_feature[i]
                support_feature_i = self.emb_fusion(
                    support_image_feature_i.unsqueeze(0),
                    support_text_feature_i.unsqueeze(0),
                    mode=fusion_method
                ).squeeze(0)
                support_image_feature[i] = support_image_feature_i

                inner_loss = F.cross_entropy(support_feature_i, support_labels[i])
                self.feature_extractor.zero_grad()
                params = gradient_update_parameters(self.feature_extractor,
                                                    inner_loss,
                                                    # step_size=1,
                                                    first_order=False)
                query_image_feature_i = self.feature_extractor(query_image_data[i], fusion_method, params=params)
                query_text_feature_i = query_text_feature[i]
                query_feature_i = self.emb_fusion(
                    query_image_feature_i.unsqueeze(0),
                    query_text_feature_i.unsqueeze(0),
                    mode=fusion_method
                ).squeeze(0)
                query_image_feature[i] = query_image_feature_i
                outer_loss += F.cross_entropy(query_feature_i, query_labels[i])

                with torch.no_grad():
                    accuracy += get_accuracy_maml(query_image_feature_i, query_labels[i])
                i += 1

            outer_loss.div_(bs)
            accuracy.div_(bs)

        else:
            support_image_data = support_image_feature_pack
            query_image_data = query_image_feature_pack
            support_text_feature, support_text_feature_att = support_text_feature_pack
            query_text_feature, query_text_feature_att = query_text_feature_pack
            device = torch.device('cuda' if support_image_data.is_cuda else 'cpu')
            self.t = self.t.to(device)
            self.matching_loss_coeff = self.matching_loss_coeff.to(device)
            self.fusion_fc = self.fusion_fc.to(device)

            support_image_feature = torch.zeros(support_text_feature.shape).to(device)
            query_image_feature = torch.zeros(query_text_feature.shape).to(device)
            outer_loss = torch.tensor(0., device=device)
            accuracy = torch.tensor(0., device=device)
            bs = support_image_data.shape[0]
            i = 0
            while i < bs:
                support_image_feature_i, support_image_feature_att_i = self.feature_extractor(support_image_data[i], fusion_method)
                support_text_feature_i, support_text_feature_att_i = support_text_feature[i], support_text_feature_att[i]
                support_feature_i = self.emb_fusion(
                    support_image_feature_att_i.unsqueeze(0),
                    support_text_feature_att_i.unsqueeze(0),
                    mode=fusion_method
                ).squeeze(0)
                support_image_feature[i] = support_image_feature_i

                inner_loss = F.cross_entropy(support_feature_i, support_labels[i])
                self.feature_extractor.zero_grad()
                params = gradient_update_parameters(self.feature_extractor,
                                                    inner_loss,
                                                    # step_size=1,
                                                    first_order=False)

                query_image_feature_i, query_image_feature_att_i = self.feature_extractor(query_image_data[i], fusion_method, params)
                query_text_feature_i, query_text_feature_att_i = query_text_feature[i], query_text_feature_att[i]

                query_feature_i = self.emb_fusion(
                    query_image_feature_att_i.unsqueeze(0),
                    query_text_feature_att_i.unsqueeze(0),
                    mode=fusion_method
                ).squeeze(0)

                query_image_feature[i] = query_image_feature_i
                outer_loss += F.cross_entropy(query_feature_i, query_labels[i])

                with torch.no_grad():
                    accuracy += get_accuracy_maml(query_feature_i, query_labels[i])
                i += 1

            outer_loss.div_(bs)
            accuracy.div_(bs)

        # (bs, num_way * (num_shot + num_query), emb_size)
        all_image_feature = torch.cat([support_image_feature, query_image_feature], dim=1)
        # (bs, num_way * (num_shot + num_query), emb_size)
        all_text_feature = torch.cat([support_text_feature, query_text_feature], dim=1)
        # # (bs, num_way * (num_shot + num_query), )
        # all_label = torch.cat([support_labels, query_labels], dim=1)

        # l2 norm for clip loss
        all_text_feature = F.normalize(all_text_feature, dim=-1)
        all_image_feature = F.normalize(all_image_feature, dim=-1)

        def clip_loss(all_image_feature, all_text_feature):
            #  clip the temperature from 0 to log100
            self.t.data = torch.clamp(self.t.data, 0, 4.605170185988092)
            # torch.clamp(self.model.module.logit_scale.data, 0, 4.6052)
            assert all_image_feature.shape[0] == all_text_feature.shape[0]
            bs = all_image_feature.shape[0]
            i = 0
            total_loss = 0
            while i < bs:
                # (80, 128)
                img_feat = all_image_feature[i]
                # (80, 128)
                text_feat = all_text_feature[i]
                scale = torch.exp(self.t)
                logits_per_image = scale * img_feat @ text_feat.t()
                logits_per_text = scale * text_feat @ img_feat.t()
                labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
                image_loss = F.cross_entropy(logits_per_image, labels)
                text_loss = F.cross_entropy(logits_per_text, labels)
                contrastive_loss = (image_loss + text_loss) / 2
                total_loss += contrastive_loss
                i += 1
            return total_loss / bs

        matching_loss = clip_loss(all_image_feature, all_text_feature)

        loss = outer_loss + matching_loss

        if is_train:
            return accuracy, loss
        else:
            return accuracy, loss.detach().cpu()
