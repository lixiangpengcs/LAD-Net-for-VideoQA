import torch.nn as nn
from models.language_model import WordEmbedding, QuestionEmbedding
from models.classifier import SimpleClassifier
from models.fc import FCNet
import torch
from torch.autograd import Variable
import numpy as np
from models.attention import NewAttention, CoAttention, paraAttention, ParalCoAttention
from models.Layer import MultiHeadAttention, PositionwiseFeedForward
import time
from models.fusion import MutanFusion

class TransModel(nn.Module):
    def __init__(self, model_name, q_proj, co_atts, q_fusion_att, v_fusion_att, context_gate, classifier, num_choice=5):
        super(TransModel, self).__init__()
        self.model_name = model_name
        self.q_proj = q_proj
        self.co_atts = co_atts
        self.q_fusion = q_fusion_att
        self.v_fusion = v_fusion_att
        self.context_gate = context_gate
        self.classifier = classifier
        self.num_choice = num_choice

    def forward(self, v, q, q_embed, labels):
        q_embed = self.q_proj(q_embed)  # bs x 35 x 512
        v_att, q_att = v, q_embed
        disagreement_loss = 0
        for i in range(len(self.co_atts)):
            v_att, q_att, dis_loss = self.co_atts[i](v_att, q_att)
            disagreement_loss += dis_loss
        v_re, _ = self.v_fusion(v, v_att)
        q_re, _ = self.q_fusion(q_embed, q_att)  # v_att: b x 512
        logits = torch.cat([v_re, q_re], dim=1)  # bs x 1024
        logits = torch.sigmoid(self.context_gate(logits)) * logits
        pred = self.classifier(logits)

        pred = pred.view(int(q.shape[0] / self.num_choice), self.num_choice)
        return pred, disagreement_loss

    def evaluate(self, dataloader):
        score = 0
        num_data = 0
        for v, q, q_embed, a in iter(dataloader):
            q = np.array(q)
            q = torch.from_numpy(q.reshape(-1, q.shape[-1]))
            q_embed = np.array(q_embed)
            q_embed = torch.from_numpy(q_embed.reshape(-1, q_embed.shape[-2], q_embed.shape[-1]))
            q_embed = Variable(q_embed).cuda()
            v = np.array(v)
            v = np.tile(v, [1, self.num_choice]).reshape(-1, v.shape[-2], v.shape[-1])
            v = Variable(torch.from_numpy(v).cuda())
            q = Variable(q.cuda())
            pred, _ = self.forward(v, q, q_embed, None)
            batch_score = compute_score_with_logits(pred, a.cuda())
            score += batch_score
            num_data += pred.size(0)

        score = float(score) / len(dataloader.dataset)
        return score


def build_temporalAtt(task_name, dataset, params):
    num_hid = params['num_hid']
    q_proj = FCNet([768, num_hid])
    bi_num_hid = num_hid * 2
    co_att = CoAttention(dataset.v_dim, num_hid, bi_num_hid)
    v_fusion_att = paraAttention(fuse_dim=dataset.v_dim, glimpses=params['sub_nums'], inputs_dim=dataset.v_dim, att_dim=num_hid)
    q_fusion_att = paraAttention(fuse_dim=num_hid, glimpses=params['sub_nums'], inputs_dim=num_hid, att_dim=num_hid)
    classifier = SimpleClassifier(
        2 * num_hid, num_hid * 2, 1, 0.5)
    return TransModel(task_name, q_proj, co_att, q_fusion_att, v_fusion_att, classifier)

def build_ParalCoAtt(task_name, dataset, params):
    # w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    # q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    num_hid = params['num_hid']
    q_proj = FCNet([768, num_hid])
    bi_num_hid = num_hid*2
    co_atts = nn.ModuleList([ParalCoAttention(dataset.v_dim, num_hid, num_hid, inter_dims=params['scale'], R=len(params['scale'])) for _ in range(params['reasonSteps'])])
    v_fusion_att = paraAttention(fuse_dim=dataset.v_dim, glimpses=params['sub_nums'], inputs_dim=dataset.v_dim, att_dim=num_hid)
    q_fusion_att = paraAttention(fuse_dim=num_hid, glimpses=params['sub_nums'], inputs_dim=num_hid, att_dim=num_hid)
    context_gate = FCNet([bi_num_hid, bi_num_hid])
    classifier = SimpleClassifier(
        bi_num_hid, num_hid * 2, 1, 0.5)
    return TransModel(task_name, q_proj, co_atts, q_fusion_att, v_fusion_att, context_gate, classifier)

def compute_score_with_logits(logits, labels):
    # logits = logits.view(labels.size(0), 5)
    logits = torch.max(logits, 1)[1]
    pred_y = logits.data.cpu().numpy().squeeze()
    target_y = labels.cpu().numpy().squeeze()
    scores = sum(pred_y == target_y)
    return scores
