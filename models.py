import torch.nn as nn
import torch
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from utils import get_accuracy
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, num_way, num_shot, num_query, emb_size):
        super(ProtoNet, self).__init__()
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query
        self.pred_loss = nn.CrossEntropyLoss()
        self.img_matching_loss = nn.CrossEntropyLoss()
        self.text_matching_loss = nn.CrossEntropyLoss()

        self.t = nn.Parameter(torch.Tensor(1))
        self.matching_loss_coeff = nn.Parameter(torch.Tensor(1))
        self.fusion_fc = nn.Linear(2 * emb_size, emb_size)

    def emb_fusion(self, support_image_feature, support_text_feature, query_image_feature, query_text_feature, mode="mean"):
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
            pass
        else:
            "please specify a fusion method"
        return support_feature, query_feature

    def forward(self, backbone_output, support_labels, query_labels):

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
            mode="fc"
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

        # cos_sim = torch.bmm(all_image_feature, all_text_feature.permute(0, 2, 1))
        # cos_sim_with_temp = cos_sim * torch.exp(self.t)
        #
        # # matching loss
        # def contrastive_loss(logits, dim):
        #     neg_ce = torch.diag(torch.nn.functional.log_softmax(logits, dim=dim))
        #     return -neg_ce.mean()
        #
        # def clip_loss(similarity):
        #     total_loss = 0
        #     for each_similarity_matrix in similarity:
        #         caption_loss = contrastive_loss(each_similarity_matrix, dim=0)
        #         image_loss = contrastive_loss(each_similarity_matrix, dim=1)
        #         cl = (caption_loss + image_loss) / 2.0
        #         total_loss += cl
        #     return total_loss / len(similarity)
        #
        # loss_img = clip_loss(cos_sim_with_temp)
        # loss_text = clip_loss(cos_sim_with_temp)
        # matching_loss = (loss_img + loss_text) / 2
        # # loss = cls_loss + self.matching_loss_coeff * matching_loss
        # loss = cls_loss + matching_loss

        # matching loss
        def contrastive_loss(logits, dim):
            neg_ce = torch.diag(torch.nn.functional.log_softmax(logits, dim=dim))
            return -neg_ce.mean()

        def clip_loss(all_image_feature, all_text_feature):
            assert all_image_feature.shape[0] == all_text_feature.shape[0]
            bs = all_image_feature.shape[0]
            i = 0
            total_loss = 0
            while i < bs:
                img_feat = all_image_feature[i]
                text_feat = all_text_feature[i]
                cos_sim = torch.matmul(img_feat, text_feat.T)
                cos_sim_with_temp = cos_sim * torch.exp(self.t)
                caption_loss = contrastive_loss(cos_sim_with_temp, dim=0)
                image_loss = contrastive_loss(cos_sim_with_temp, dim=1)
                cl = (caption_loss + image_loss) / 2.0
                total_loss += cl
                i += 1
            return total_loss / bs
        matching_loss = clip_loss(all_image_feature, all_text_feature)
        # loss = cls_loss + self.matching_loss_coeff * matching_loss
        loss = cls_loss + matching_loss
        with torch.no_grad():
            accuracy = get_accuracy(prototypes, query_feature, query_labels)
        return accuracy, loss
