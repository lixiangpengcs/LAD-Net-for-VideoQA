import torch.nn as nn
from models.language_model import WordEmbedding, QuestionEmbedding
from models.classifier import SimpleClassifier
from models.fc import FCNet
import torch
from torch.autograd import Variable
import numpy as np
from models.attention import NewAttention, CoAttention, paraAttention, ParalCoAttention
from models.fusion import MutanFusion, ParalFusion

class CountModel(nn.Module):
    def __init__(self, model_name, q_proj, co_atts, q_fusion_att, v_fusion_att, context_gate, classifier, num_choice=1):
        super(CountModel, self).__init__()
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
            v_att, q_att, dis_loss = self.co_atts[i](v_att,q_att)  # input v: bs x 36 x 1024    q: bs x 35 x 1024  outputs: as inputs
            disagreement_loss += dis_loss
        v_re, _ = self.v_fusion(v, v_att)
        q_re, _ = self.q_fusion(q_embed, q_att)  # v_att: b x 512
        logits = torch.cat([v_re, q_re], dim=1)  # bs x 1024
        logits = torch.sigmoid(self.context_gate(logits)) * logits
        pred = self.classifier(logits)
        pred = torch.squeeze(pred)
        return pred, disagreement_loss

    def evaluate(self, dataloader):
        score = 0
        num_data = 0

        for v, q, q_embed, a in iter(dataloader):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            q_embed = Variable(q_embed).cuda()
            logits, _ = self.forward(v, q, q_embed, None)
            logits = logits.data.cpu().numpy().squeeze()
            pred = np.clip(np.round(logits), a_min=1, a_max=10)
            batch_score = compute_score_with_logits(pred, a.cuda())
            score += batch_score
            # num_data += pred.size(0)

        score = float(score) / len(dataloader.dataset)
        return score

    def sample(self, dataset, dataloader):
        import json
        j = 0
        results =[]
        for v,q,a in iter(dataloader):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            pred = self.forward(v, q, None)
            logits = pred.data.cpu().numpy().squeeze()
            pred_label = int(np.clip(np.round(logits), a_min=1, a_max=10))
            # pred_label.squeeze_()
            key = dataset.entries[j]['key']
            question = dataset.entries[j]['question']
            gruth_label = int(dataset.entries[j]['labels'].numpy()[0])
            results.append(
                {'key':key,
                 'question':question,
                 'pred_answer': pred_label,
                 'gt_answer': gruth_label,
                 'j': j
                 }
            )
            j = j + 1
        json.dump(results, open('Count.json', 'w'))


def build_baseline(task_name, dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_emb = QuestionEmbedding(dataset.v_dim, num_hid, 1, False, 0.0)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([num_hid, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, 1, 0.5)
    return CountModel(task_name, w_emb, q_emb, v_emb, q_net, v_net, classifier)

def build_temporalAtt(task_name, dataset, params):
    num_hid = params['num_hid']
    q_proj = FCNet([768, num_hid])
    bi_num_hid = num_hid * 2
    co_att = CoAttention(dataset.v_dim, num_hid, bi_num_hid)
    v_fusion_att = paraAttention(fuse_dim=dataset.v_dim, glimpses=params['sub_nums'], inputs_dim=dataset.v_dim, att_dim=num_hid)
    q_fusion_att = paraAttention(fuse_dim=num_hid, glimpses=params['sub_nums'], inputs_dim=num_hid, att_dim=num_hid)
    classifier = SimpleClassifier(
        num_hid*2, num_hid * 2, 1, 0.5)
    return CountModel(task_name, q_proj, co_att, q_fusion_att, v_fusion_att, classifier)

def build_ParalCoAtt(task_name, dataset, params):
    # w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    # q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    num_hid = params['num_hid']
    q_proj = FCNet([768, num_hid])
    bi_num_hid = num_hid*2
    co_atts = nn.ModuleList(
        [ParalCoAttention(dataset.v_dim, num_hid, num_hid, inter_dims=params['scale'], R=len(params['scale'])) for _ in
         range(params['reasonSteps'])])
    # co_att = ParalCoAttention(dataset.v_dim, num_hid, num_hid, inter_dims=params['scale'], R=params['glimpse'])
    v_fusion_att = paraAttention(fuse_dim=dataset.v_dim, glimpses=params['sub_nums'], inputs_dim=dataset.v_dim, att_dim=num_hid)
    q_fusion_att = paraAttention(fuse_dim=num_hid, glimpses=params['sub_nums'], inputs_dim=num_hid, att_dim=num_hid)
    context_gate = FCNet([bi_num_hid, bi_num_hid])
    classifier = SimpleClassifier(
        bi_num_hid, num_hid * 2, 1, 0.5)
    return CountModel(task_name, q_proj, co_atts, q_fusion_att, v_fusion_att, context_gate, classifier)

def compute_score_with_logits(logits, labels):
    # logits = torch.max(logits, 1)[1]=
    pred_y = logits#.data.cpu().numpy().squeeze()
    target_y = labels.cpu().numpy()
    score = sum(np.square(pred_y-target_y))
    return score
