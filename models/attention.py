import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from models.fc import FCNet
import torch.nn.functional as F
from models.fusion import Linear
from itertools import combinations
eps = 0.00001
def cosine_distance(a, b): # input: a:  bs x len x dim
    dot_mul = torch.matmul(a, b.transpose(2,1))
    a_norm = torch.norm(a, dim=2) # bs x len
    b_norm = torch.norm(b, dim=2) # bs x len
    a_norm_v = a_norm.view(a_norm.size(0), a_norm.size(1), 1) # bs x len x 1
    b_norm_v = b_norm.view(b_norm.size(0), 1, b_norm.size(1)) # bs x 1 x len
    low_mul = torch.matmul(a_norm_v, b_norm_v)  # bs x len x len
    cos_dis = dot_mul / (low_mul+eps)
    dis_loss = torch.sum(cos_dis) / (a.size(0) * a.size(1) * b.size(1))

    return dis_loss

def disagreement_loss(q_atts, v_atts, R):
    combs = list(combinations(list(range(R)), 2))
    q_dis_loss = 0
    v_dis_loss = 0
    for comb in combs:
        q_dis_loss += cosine_distance(q_atts[comb[0]], q_atts[comb[1]])
        v_dis_loss += cosine_distance(v_atts[comb[0]], v_atts[comb[1]])
    return (q_dis_loss + v_dis_loss)/(R*2)
class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class CoAttention(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, dropout=[.2,.5]):
        super(CoAttention, self).__init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.hid_dim = hid_dim
        act = "ReLU"
        self.v_net = FCNet([v_dim,  self.hid_dim], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, self.hid_dim], act=act, dropout=dropout[0])


    def forward(self, v, q, v_mask=True):
        v_fc = self.v_net(v)   # bs x frame_num x dim
        q_fc = self.q_net(q)   # bs x q_len x dim
        aff_mat = torch.matmul(v_fc, q_fc.transpose(1, 2))   # bs x frame_len x q_len
        v_att = nn.functional.softmax(aff_mat, dim=1)[:,:,0].unsqueeze(2) # bs x frame_len
        q_att = nn.functional.softmax(aff_mat, dim=2)[:,0,:].unsqueeze(2) # bs x q_len
        v_attend = (v * v_att) + v # bs x 36 x 2048
        q_attend = (q * q_att) + q # bs x 35 x 512
        return v_attend, q_attend

class ParalCoAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, inter_dims, R, dropout=[.2,.5]):
        super(ParalCoAttention, self).__init__()
        self.R = R
        self.num_dim = num_hid
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.inter_dims = inter_dims
        act = "ReLU"
        assert len(self.inter_dims) == self.R
        self.list_v_net = nn.ModuleList(
            [FCNet([v_dim, inter_dim], act=act, dropout=dropout[0]) for inter_dim in self.inter_dims]
        )
        self.list_q_net = nn.ModuleList(
            [FCNet([q_dim, inter_dim], act=act, dropout=dropout[0]) for inter_dim in self.inter_dims]
        )
        assert len(self.list_v_net) == self.R
        assert len(self.list_q_net) == self.R

    def forward(self, v, q, v_mask=True):
        q_states = 0
        v_states = 0
        v_atts = []
        q_atts = []
        for i in range(self.R):
            h_v = self.list_v_net[i](v)
            h_q = self.list_q_net[i](q)
            aff_mat = torch.matmul(h_v, h_q.transpose(1, 2))
            v_att = nn.functional.softmax(aff_mat, dim=1)[:, :, 0].unsqueeze(2)
            q_att = nn.functional.softmax(aff_mat, dim=2)[:, 0, :].unsqueeze(2)
            v_attend = (v * v_att)   # bs x dim
            q_attend = (q * q_att)
            v_atts.append(v_attend)
            q_atts.append(q_attend)
            q_states += q_attend
            v_states += v_attend
        q_states = q_states + q
        v_states = v_states + v
        loss = disagreement_loss(q_atts, v_atts, self.R)
        return v_states, q_states, loss

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)

class MyConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None, p=None, af=None,
                 dim=None):
        super(MyConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.p = p
        self.af = af
        self.dim = dim
        if seed:
            torch.manual_seed(seed)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=1, groups=1, bias=True)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError('[error] putils.Conv1d(%s, %s, %s, %s): input_dim (%s) should equal to 3' %
                             (self.in_channels, self.out_channels, self.kernel_size, self.stride, x.dim()))
        # x: b*49*512
        if self.p:
            x = F.dropout(x, p=self.p, training=self.training)
        x = x.transpose(1, 2)  # b*512*49
        x = self.conv(x)  # b*450*49
        x = x.transpose(1, 2)  # b*49*450

        if self.af:
            if self.af == 'softmax':
                x = getattr(F, self.af)(x, dim=self.dim)
            else:
                x = getattr(F, self.af)(x)
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, seed=None, p=None, af=None, dim=None): # p: dropout
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.af = af
        self.dim = dim
        if seed:
            torch.manual_seed(seed)
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        if x.size()[-1] != self.in_features:
            raise ValueError(
                '[error] putils.Linear(%s, %s): last dimension of input(%s) should equal to in_features(%s)' %
                (self.in_features, self.out_features, x.size(-1), self.in_features))
        if self.p:
            x = F.dropout(x, p=self.p, training=self.training)
        x = self.linear(x)
        if self.af:
            if self.af == 'softmax':
                x = getattr(F, self.af)(x, dim=self.dim)
            else:
                x = getattr(F, self.af)(x)
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class paraAttention(nn.Module):
    def __init__(self, fuse_dim, glimpses, inputs_dim, att_dim, seed=None, af='tanh'):
        super(paraAttention, self).__init__()
        assert att_dim % glimpses == 0
        self.glimpses = glimpses
        self.inputs_dim = inputs_dim
        self.att_dim = att_dim
        self.conv_att = MyConv1d(fuse_dim, glimpses, 1, 1, seed=seed, p=0.5, af='softmax', dim=1)  # (510, 2, 1, 1)
        self.list_linear_v_fusion = nn.ModuleList(
            [MyLinear(inputs_dim, int(att_dim / glimpses), p=0.5, af=af) for _ in range(glimpses)])  # (2048, 155) * 4
        self.af = af

    def forward(self, inputs, fuse): # bs x 36 x 2048    bs x 36 x 2048
        b = inputs.size(0)
        n = inputs.size(1)
        x_att = self.conv_att(fuse) # bs x 36 x 4
        # x_att = nn.functional.softmax(x_att, dim=1)
        tmp = torch.matmul(x_att.transpose(1, 2), inputs)  # b*4*2048

        list_v_att = [e.squeeze() for e in torch.split(tmp, 1, dim=1)]  # b*2048, b*2048, b*2048, b*2048

        list_att = torch.split(x_att, 1, dim=2) # [bx36, bx36, bx36, bx36]

        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = self.list_linear_v_fusion[glimpse_id](x_v_att)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1) #
        return x_v, list_att

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)