import torch.nn as nn
import torch
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from utils import get_accuracy
import torch.nn.functional as F
import math
import numpy as np


class Attention(torch.nn.Module):

    def __init__(self, q_dim, k_dim, v_dim, h_dim):
        super().__init__()
        self.q = nn.Linear(q_dim, h_dim)
        self.k = nn.Linear(k_dim, h_dim)
        self.v = nn.Linear(v_dim, h_dim)

    def similarity(self, attention_type, query, key):
        """
        Similarity Function of Attention
        :param attention_type: additive, scale dot, mlp
        :param query: (bs, 80, 128), text
        :param key: (bs, 80, 128), image
        :return: (bs, 80, 80)
        """

        scores = torch.bmm(query, key.permute(0, 2, 1)) / math.sqrt(key.shape[-1])

        return scores

    def forward(self, query, key, value):
        """
        https://arxiv.org/pdf/1706.03762.pdf
        :param query: (bs, 80, 128), text
        :param key: (bs, 80, 128), image
        :param value: (bs, 80, 128), image
        :return: (bs, 80, 128)
        """        
        
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)
        scores = self.similarity("scaled_dot_product", query, key)
        att_map = F.softmax(scores, dim=-1)
        context = torch.bmm(att_map, value)

        return context
    

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
            support_feature = self.attention(support_text_feature, support_image_feature, support_image_feature)
            query_feature = self.attention(query_text_feature, query_image_feature, query_image_feature)
        else:
            "please specify a fusion method"
            exit()
        return support_feature, query_feature

    def forward(self, backbone_output, support_labels, query_labels, fusion_method="mean"):

        # support_image_feature: (bs, num_way * num_shot, emb_size)
        # query_image_feature: (bs, num_way * num_query, emb_size)
        # support_text_feature: (bs, num_way * num_shot, emb_size)
        # query_text_feature: (bs, num_way * num_query, emb_size)

        support_image_feature, query_image_feature, support_text_feature, query_text_feature = backbone_output
        support_text_feature = F.normalize(support_text_feature, dim=-1)
        query_image_feature = F.normalize(query_image_feature, dim=-1)
        query_text_feature = F.normalize(query_text_feature, dim=-1)
        support_image_feature = F.normalize(support_image_feature, dim=-1)

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
        # prototypes: (bs, num_way, emb_size)
        prototypes = get_prototypes(support_feature, support_labels, self.num_way)
        cls_loss = prototypical_loss(prototypes, query_feature, query_labels)

        # (bs, num_way * (num_shot + num_query), emb_size)
        all_image_feature = torch.cat([support_image_feature, query_image_feature], dim=1)
        # (bs, num_way * (num_shot + num_query), emb_size)
        all_text_feature = torch.cat([support_text_feature, query_text_feature], dim=1)
        # (bs, num_way * (num_shot + num_query), )
        all_label = torch.cat([support_labels, query_labels], dim=1)

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

                # cos_sim = torch.matmul(img_feat, text_feat.T)
                # cos_sim_with_temp = cos_sim * torch.exp(self.t)
                # image_loss = contrastive_loss(cos_sim_with_temp, dim=0)
                # caption_loss = contrastive_loss(cos_sim_with_temp, dim=1)
                # cl = (image_loss + caption_loss) / 2.0
                # # matching loss
                # def contrastive_loss(logits, dim):
                #     neg_ce = torch.diag(torch.nn.functional.log_softmax(logits, dim=dim))
                #     return -neg_ce.mean()

                # cosine similarity as logits
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
