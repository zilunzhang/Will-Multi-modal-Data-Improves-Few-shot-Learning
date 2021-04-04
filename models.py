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

        support_feature = (support_image_feature.norm(2, dim=2, keepdim=True) + support_text_feature.norm(2, dim=2, keepdim=True)) / 2
        query_feature = (query_image_feature.norm(2, dim=2, keepdim=True) + query_text_feature.norm(2, dim=2, keepdim=True)) / 2

        # prototypes: (bs, num_way, emb_size)
        prototypes = get_prototypes(support_feature, support_labels, self.num_way)
        cls_loss = prototypical_loss(prototypes, query_feature, query_labels)

        # (bs, num_way * (num_shot + num_query), emb_size)
        all_image_feature = torch.cat([support_image_feature, query_image_feature], dim=1)
        # (bs, num_way * (num_shot + num_query), emb_size)
        all_text_feature = torch.cat([support_text_feature, query_text_feature], dim=1)
        # (bs, num_way * (num_shot + num_query), )
        all_label = torch.cat([support_labels, query_labels], dim=1)

        # # (bs * num_way * (num_shot + num_query), emb_size)
        # all_image_feature = torch.reshape(all_image_feature, (all_image_feature.shape[0] * all_image_feature.shape[1], -1))
        # # (bs * num_way * (num_shot + num_query), emb_size)
        # all_text_feature = torch.reshape(all_text_feature, (all_text_feature.shape[0] * all_text_feature.shape[1], -1))
        # # (bs * num_way * (num_shot + num_query), )
        # all_label = torch.reshape(all_label, (all_label.shape[0] * all_label.shape[1], ))

        # self.t = nn.Parameter(torch.Tensor(len(all_label), 1))
        # # (bs * num_way * (num_shot + num_query), bs * num_way * (num_shot + num_query))
        # cos_sim = torch.dot(all_image_feature, all_text_feature.T)
        # cos_sim_with_temp = cos_sim * torch.exp(self.t)

        device = torch.device('cuda' if support_feature.is_cuda else 'cpu')
        self.t = self.t.to(device)
        self.matching_loss_coeff = self.matching_loss_coeff.to(device)

        cos_sim = torch.bmm(all_image_feature, all_text_feature.permute(0, 2, 1))
        cos_sim_with_temp = cos_sim * torch.exp(self.t)
        # matching loss function
        def contrastive_loss(logits, dim):
            neg_ce = torch.diag(torch.nn.functional.log_softmax(logits, dim=dim))
            return -neg_ce.mean()

        def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
            total_loss = 0
            for each_similarity_matrix in similarity:
                caption_loss = contrastive_loss(each_similarity_matrix, dim=0)
                image_loss = contrastive_loss(each_similarity_matrix, dim=1)
                cl = (caption_loss + image_loss) / 2.0
                total_loss += cl
            return total_loss / len(similarity)

        loss_img =  clip_loss(cos_sim_with_temp)
        loss_text =  clip_loss(cos_sim_with_temp)
        matching_loss = (loss_img + loss_text) / 2

        # loss = cls_loss + self.matching_loss_coeff * matching_loss

        loss = cls_loss + 0.5 * matching_loss

        with torch.no_grad():
            accuracy = get_accuracy(prototypes, query_feature, query_labels)

        return accuracy, loss
