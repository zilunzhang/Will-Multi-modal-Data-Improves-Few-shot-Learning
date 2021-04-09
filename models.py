import torch.nn as nn
import torch
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from utils import get_accuracy
import torch.nn.functional as F
import math
import numpy as np
    

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
        else:
            "please specify a fusion method"
            exit()
        return support_feature, query_feature

    def forward(self, backbone_output, support_labels, query_labels, fusion_method="mean"):

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
        # (bs, num_way * (num_shot + num_query), )
        all_label = torch.cat([support_labels, query_labels], dim=1)

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
            nn.Linear(q_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def similarity(self, attention_type, query, key, choice="mixed"):
        """
        Similarity Function of Attention
        :param attention_type: additive, scale dot, mlp
        :param query: (bs, 5, 10, 128), text
        :param key: (bs, 5, 10, 128), image
        :return: (bs, 5, 10, 128)
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
                # (5, 10, 10)
                scores = torch.bmm(query[i], key[i].permute(0, 2, 1)) / math.sqrt(key[i].shape[-1])
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
        https://arxiv.org/pdf/1706.03762.pdf
        :param query: (bs, 5, 10, 128), text
        :param key: (bs, 5, 10, 128), image
        :param value: (bs, 5, 10, 128), image
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
        # scores = self.similarity("scaled_dot_product", query, key)
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
        scores = self.similarity("scaled_dot_product", query, key, choice="bs")
        att_map = F.softmax(scores, dim=-1)

        # loop through bs axis
        num_bs = value.shape[0]
        # (bs, 5, 10, 128)
        context = torch.zeros(value.shape[0], value.shape[1], value.shape[2], value.shape[3]).to(device)
        i = 0
        while i < num_bs:
            # (5, 10, 128)
            context[i] = torch.bmm(att_map[i], value[i])
            i += 1

        # (bs, 5, 10, 1)
        att_reduce_weight = self.reduce_att(context)
        # (bs, 5, 128, 1)
        attended_content = torch.zeros(context.shape[0], context.shape[1], context.shape[3], att_reduce_weight.shape[3]).to(device)
        i = 0
        while i < num_bs:
            # (5, 128, 10) * (5, 10, 1) -> (5, 128, 1)
            attended_content[i] = torch.bmm(context[i].permute(0, 2, 1), att_reduce_weight[i])
            i += 1
    
        return attended_content.squeeze(-1)