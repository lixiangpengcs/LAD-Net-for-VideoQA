from utils import *
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.attention import *
from torch.nn.parameter import Parameter
# import spacy

print('[info] putils: Enhanced Deep Learning Putils loaded successfully, enjoy it!')


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, seed=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if seed:
            torch.manual_seed(seed)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        if x.size()[-1] != self.in_features:
            raise ValueError(
                '[error] putils.Linear(%s, %s): last dimension of input(%s) should equal to in_features(%s)' %
                (self.in_features, self.out_features, x.size(-1), self.in_features))
        return self.linear(x)

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if seed:
            torch.manual_seed(seed)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=1, groups=1, bias=True)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('[error] putils.Conv2d(%s, %s, %s, %s): input_dim (%s) should equal to 4' %
                             (self.in_channels, self.out_channels, self.kernel_size, self.stride, x.dim()))

        # x: b*7*7*512
        x = x.transpose(2, 3).transpose(1, 2)  # b*512*7*7
        x = self.conv(x)  # b*450*7*7
        x = x.transpose(1, 2).transpose(2, 3)  # b*7*7*450
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if seed:
            torch.manual_seed(seed)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=1, groups=1, bias=True)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError('[error] putils.Conv1d(%s, %s, %s, %s): input_dim (%s) should equal to 3' %
                             (self.in_channels, self.out_channels, self.kernel_size, self.stride, x.dim()))
        # x: b*49*512
        x = x.transpose(1, 2)  # b*512*49
        x = self.conv(x)  # b*450*49
        x = x.transpose(1, 2)  # b*49*450
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


def bmatmul(inputs1, inputs2):
    b = inputs1.size()[0]
    m = []
    for i in range(b):
        m.append(torch.matmul(inputs1[i], inputs2[i]))
    outputs = torch.stack(m, dim=0)
    return outputs


def bmul(inputs1, inputs2): # bs x 36 x 512    bs x 512
    b = inputs1.size()[0]
    m = []
    for i in range(b):
        m.append(inputs1[i] * inputs2[i])   # bs x 36 x 512
    outputs = torch.stack(m, dim=0)
    return outputs

def softatt(inputs1, inputs2):
    inputs2 = inputs2.unsqueeze(2) # bs x 512 x 1
    logits = torch.matmul(inputs1, inputs2) # bs x 36 x 1
    att = F.softmax(logits, dim=1) # bs x 36 x 1
    pred = inputs1 * att
    # pred = pred + inputs1
    return pred

def bmul3(inputs1, inputs2, inputs3):
    b = inputs1.size()[0]
    m = []
    for i in range(b):
        m.append(inputs1[i] * inputs2[i] * inputs3[i])
    outputs = torch.stack(m, dim=0)
    return outputs


def badd(inputs, inputs2):
    b = inputs.size()[0]
    m = []
    for i in range(b):
        m.append(inputs[i] + inputs2[i])
    outputs = torch.stack(m, dim=0)
    return outputs


class RelationFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):
        '''
        Note: can only fusion two inputs with batch.

        inputs1: b*..*input_dim1
        inputs2: b*...*input_dim2
        outputs: b*...*hidden_dim

        调整为最长，最后元素调整为hidden_dim

        :param input_dim1:
        :param input_dim2:
        :param hidden_dim:
        :param R: Do element-wise product R times.
        :param seed: random seed.
        '''
        super(RelationFusion, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear_hv = nn.ModuleList(
            [nn.Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear_hq = nn.ModuleList(
            [nn.Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, inputs1, inputs2):
        x_mm = []
        for i in range(self.R):
            h1 = self.list_linear_hv[i](inputs1)
            h2 = self.list_linear_hq[i](inputs2)
            x_mm.append(torch.bmm(h1, h2.transpose(1, 2)))
        x_mm = torch.stack(x_mm, dim=1)
        return x_mm

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MutanFusionOld(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):
        '''
        Note: can only fusion two inputs with batch.

        inputs1: b*..*input_dim1
        inputs2: b*...*input_dim2
        outputs: b*...*hidden_dim

        调整为最长，最后元素调整为hidden_dim

        :param input_dim1:
        :param input_dim2:
        :param hidden_dim:
        :param R: Do element-wise product R times.
        :param seed: random seed.
        '''
        super(MutanFusionOld, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear_hv = nn.ModuleList(
            [nn.Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear_hq = nn.ModuleList(
            [nn.Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, inputs1, inputs2):
        x_mm = []
        for i in range(self.R):
            h1 = self.list_linear_hv[i](inputs1)
            h2 = self.list_linear_hq[i](inputs2)
            x_mm.append(bmul(h1, h2))
        x_mm = torch.stack(x_mm, dim=1).sum(1)
        return x_mm

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


def normal(shape, scale=0.05):
    tensor = torch.FloatTensor(*shape)
    tensor.normal_(mean=0.0, std=scale)
    return tensor

def get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def glorot_normal(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / (fan_in + fan_out))
    return normal(shape, s)

class MutanFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):  # 1024, 1024, 1024 , 2

        super(MutanFusion, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear1 = nn.ModuleList(
            [Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear2 = nn.ModuleList(
            [Linear(input_dim2, hidden_dim) for _ in range(R)])

        self.w = Parameter(glorot_normal((R, )))
        self.b = Parameter(glorot_normal((R, )))

        # self.sen = SEN(512, 4)

    def forward(self, inputs1, inputs2):
        # total = 0
        outs = []
        for i in range(self.R):
            h1 = self.list_linear1[i](inputs1) #  bs x 36 x 512
            h2 = self.list_linear2[i](inputs2) #  bs x 512
            # logit += bmul(h1, h2) # bs x 36 x 512
            logit = softatt(h1, h2)
            w = self.w[i].unsqueeze(0).expand(logit.size(0), logit.size(-1))
            b = self.b[i].unsqueeze(0).expand(logit.size(0), logit.size(-1))
            o = logit * w + b

        return total   # bs x 36 x 510

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class ParalFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):  # 1024, 1024, 1024 , 2

        super(ParalFusion, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.gnorm = np.sqrt(R)
        self.list_linear1 = nn.ModuleList(
            [Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear2 = nn.ModuleList(
            [Linear(input_dim2, hidden_dim) for _ in range(R)])

        self.w = Parameter(glorot_normal((R,)))
        self.b = Parameter(glorot_normal((R,)))

    def forward(self, inputs1, inputs2):
        # total = 0
        outs = []
        len = inputs1.size(1)
        for i in range(self.R):
            h1 = self.list_linear1[i](inputs1)  # bs x 36 x 512
            h2 = self.list_linear2[i](inputs2)  # bs x 512
            # logit += bmul(h1, h2) # bs x 36 x 512
            h2 = h2.unsqueeze(2)  # bs x 512 x 1
            logits = torch.matmul(h1, h2)  # bs x 36 x 1
            att = F.softmax(logits, dim=1)  # bs x 36 x 1
            pred = torch.sum(inputs1 * att, 1).squeeze()    # bs x 2048
            w = self.w[i].unsqueeze(0).expand(pred.size(0), pred.size(-1))
            b = self.b[i].unsqueeze(0).expand(pred.size(0), pred.size(-1))
            o = pred * w + b
            norm = torch.norm(o, 2, -1, keepdim=True).expand_as(o)
            o = o/norm/self.gnorm
            # o = o.unsqueeze(1)
            outs.append(o)
        f_outs = torch.cat(outs, dim=-1).unsqueeze(1)
        f_outs = f_outs.expand(f_outs.size(0), len, f_outs.size(2))
        return f_outs  # bs x 36 x 510

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MutanFusion2D(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):
        super(MutanFusion2D, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear1 = nn.ModuleList(
            [Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear2 = nn.ModuleList(
            [Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, inputs1, inputs2):
        total = 0
        for i in range(self.R):
            h1 = self.list_linear1[i](inputs1)
            h2 = self.list_linear2[i](inputs2)
            h1 = h1.view(-1, 1, h1.size(1), h1.size(2))
            h2 = h2.view(-1, h2.size(1), 1, h2.size(2)).repeat(1, 1, h1.size(1), 1)
            total += bmul(h1, h2)
        return total

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class ATTOld(nn.Module):
    def __init__(self, fuse_dim, glimpses, inputs_dim, att_dim, af='tanh'):
        super(ATTOld, self).__init__()
        assert att_dim % glimpses == 0
        self.glimpses = glimpses
        self.inputs_dim = inputs_dim
        self.att_dim = att_dim
        self.conv_att = Conv1d(fuse_dim, glimpses, 1, 1)  # (510, 2, 1, 1)
        self.list_linear_v_fusion = nn.ModuleList(
            [nn.Linear(inputs_dim, int(att_dim / glimpses)) for _ in range(glimpses)])  # (2048, 620/n) * n
        self.af = af

    def forward(self, inputs, fuse):
        b = inputs.size(0)
        n = inputs.size(1)
        x_att = F.dropout(self.conv_att(fuse), p=0.5, training=self.training)  # b*49*2
        list_att_split = torch.split(x_att, 1, dim=2)  # (b*49*1, b*49*1)
        list_att = []  # [b*49, b*49]
        for x_att in list_att_split:
            x_att = F.softmax(x_att.squeeze(-1))  # b*49
            list_att.append(x_att)

        # Apply attention vectors to input_v
        x_v = inputs  # b*49*2048

        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(b, n, 1).expand(b, n, self.inputs_dim)
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(b, self.inputs_dim)
            list_v_att.append(x_v_att)

        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att, p=0.5, training=self.training)
            x_v = getattr(F, self.af)(self.list_linear_v_fusion[glimpse_id](x_v))
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)
        return x_v, list_att

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MutanFusionFE(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):

        super(MutanFusionFE, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear1 = nn.ModuleList(
            [Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear2 = nn.ModuleList(
            [Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, inputs1, inputs2):
        total = 0
        for i in range(self.R):
            h1 = self.list_linear1[i](inputs1)
            h2 = self.list_linear2[i](inputs2)
            total += h1*h2
        return total

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MutanFusionFNE(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R, seed=None):

        super(MutanFusionFNE, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear1 = nn.ModuleList(
            [Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear2 = nn.ModuleList(
            [Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, inputs1, inputs2):
        total = 0
        for i in range(self.R):
            h1 = self.list_linear1[i](inputs1)
            h2 = self.list_linear2[i](inputs2)
            total += (h1 * h2.view(-1, 1, self.hidden_dim))
        return total

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)

class ATT(nn.Module):
    def __init__(self, fuse_dim, glimpses, inputs_dim, att_dim, seed=None, af='tanh'):
        super(ATT, self).__init__()
        assert att_dim % glimpses == 0
        self.glimpses = glimpses
        self.inputs_dim = inputs_dim
        self.att_dim = att_dim
        self.conv_att = Conv1d(fuse_dim, glimpses, 1, 1, seed=seed)  # (510, 2, 1, 1)
        self.list_linear_v_fusion = nn.ModuleList(
            [nn.Linear(inputs_dim, int(att_dim / glimpses)) for _ in range(glimpses)])  # (2048, 620/n) * n
        self.af = af

    def forward(self, inputs, fuse):
        b = inputs.size(0)
        n = inputs.size(1)
        x_att = F.dropout(self.conv_att(fuse), p=0.5, training=self.training)  # b*49*2
        list_att_split = torch.split(x_att, 1, dim=2)  # (b*49*1, b*49*1)
        list_att = []  # [b*49, b*49]
        for x_att in list_att_split:
            x_att = F.softmax(x_att.squeeze(-1), dim=-1)  # b*49
            list_att.append(x_att)

        # Apply attention vectors to input_v
        x_v = inputs  # b*49*2048

        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(b, n, 1).expand(b, n, self.inputs_dim)
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(b, self.inputs_dim)
            list_v_att.append(x_v_att)

        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att, p=0.5, training=self.training)
            x_v = getattr(F, self.af)(self.list_linear_v_fusion[glimpse_id](x_v))
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)
        return x_v, list_att

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class ATTN(nn.Module):
    def __init__(self, inputs_dim, guidance_dim, fuse_dim, att_dim, glimpses, R, seed=None, af='tanh'):
        super(ATTN, self).__init__()
        assert att_dim % glimpses == 0
        self.inputs_dim = inputs_dim
        self.guidance_dim = guidance_dim
        self.fuse_dim = fuse_dim
        self.att_dim = att_dim
        self.glimpses = glimpses
        self.R = R
        self.af = af

        self.fuse = MutanFusion(inputs_dim, guidance_dim, fuse_dim, R, seed=seed)
        self.conv_att = Conv1d(fuse_dim, glimpses, 1, 1, seed=seed)  # (510, 2, 1, 1)
        self.list_linear_v_fusion = nn.ModuleList(
            [Linear(inputs_dim, int(att_dim / glimpses), seed=seed) for _ in range(glimpses)])  # (2048, 620/n) * n

    def forward(self, inputs, guidance):
        if inputs.dim() != 3 or guidance.dim() != 2:
            raise ValueError('[error] putils.ATTN: inputs dim should be 3, guidance dim should be 2')
        fuse = self.fuse(inputs, guidance)

        b = inputs.size(0)
        n = inputs.size(1)
        x_att = F.dropout(self.conv_att(fuse), p=0.5, training=self.training)  # b*49*2
        list_att_split = torch.split(x_att, 1, dim=2)  # (b*49*1, b*49*1)
        list_att = []  # [b*49, b*49]
        for x_att in list_att_split:
            x_att = F.softmax(x_att.squeeze(-1))  # b*49
            list_att.append(x_att)

        # Apply attention vectors to input_v
        x_v = inputs  # b*49*2048

        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(b, n, 1).expand(b, n, self.inputs_dim)
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(b, self.inputs_dim)
            list_v_att.append(x_v_att)

        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att, p=0.5, training=self.training)
            x_v = getattr(F, self.af)(self.list_linear_v_fusion[glimpse_id](x_v))
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)

        if self.glimpses == 1:
            list_att = list_att[0]

        return x_v, list_att

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class EmbeddingDropout():
    def __init__(self, p=0.5):
        super(EmbeddingDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.training = True

    def forward(self, input):
        # input must be tensor
        if self.p > 0 and self.training:
            dim = input.dim()
            if dim == 1:
                input = input.view(1, -1)
            batch_size = input.size(0)
            for i in range(batch_size):
                x = np.unique(input[i].numpy())
                x = np.nonzero(x)[0]
                x = torch.from_numpy(x)
                noise = x.new().resize_as_(x)
                noise.bernoulli_(self.p)
                x = x.mul(noise)
                for value in x:
                    if value > 0:
                        mask = input[i].eq(value)
                        input[i].masked_fill_(mask, 0)
            if dim == 1:
                input = input.view(-1)

        return input


class SequentialDropout(nn.Module):
    def __init__(self, p=0.5):
        super(SequentialDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.restart = True

    def _make_noise(self, input):
        return Variable(input.data.new().resize_as_(input.data))

    def forward(self, input):
        if self.p > 0 and self.training:
            if self.restart:
                self.noise = self._make_noise(input)
                self.noise.data.bernoulli_(1 - self.p).div_(1 - self.p)
                if self.p == 1:
                    self.noise.data.fill_(0)
                self.noise = self.noise.expand_as(input)
                self.restart = False
            return input.mul(self.noise)

        return input

    def end_of_sequence(self):
        self.restart = True

    def backward(self, grad_output):
        self.end_of_sequence()
        if self.p > 0 and self.training:
            return grad_output.mul(self.noise)
        else:
            return grad_output

    def __repr__(self):
        return type(self).__name__ + '({:.4f})'.format(self.p)


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0.0, bidirectional=False, return_last=True):
        super(GRU, self).__init__()
        self.batch_first = batch_first
        self.return_last = return_last
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, bias=bias, batch_first=batch_first,
                          dropout=dropout, bidirectional=bidirectional)

    def forward(self, emb, lengths=None):
        if self.return_last:
            lengths, idx = torch.sort(lengths, dim=-1, descending=True)
            packed = pack_padded_sequence(emb[idx, :], list(lengths), batch_first=self.batch_first)
            out_packed, last = self.gru(packed)
            final = last[0][idx, :]
            return final

        else:
            o, h = self.gru(emb)
            return o


class AbstractGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size,
                 bias_ih=True, bias_hh=False):
        super(AbstractGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh

        # Modules
        self.weight_ir = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_ii = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_in = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_hr = nn.Linear(hidden_size, hidden_size, bias=bias_hh)
        self.weight_hi = nn.Linear(hidden_size, hidden_size, bias=bias_hh)
        self.weight_hn = nn.Linear(hidden_size, hidden_size, bias=bias_hh)

    def forward(self, x, hx=None):
        raise NotImplementedError


class GRUCell(AbstractGRUCell):
    def __init__(self, input_size, hidden_size,
                 bias_ih=True, bias_hh=False, af='tanh'):
        super(GRUCell, self).__init__(input_size, hidden_size,
                                      bias_ih, bias_hh)
        self.af = af

    def forward(self, x, hx=None):
        if hx is None:
            hx = Variable(x.data.new().resize_((x.size(0), self.hidden_size)).fill_(0))
        r = F.sigmoid(self.weight_ir(x) + self.weight_hr(hx))
        i = F.sigmoid(self.weight_ii(x) + self.weight_hi(hx))
        n = getattr(F, self.af)(self.weight_in(x) + r * self.weight_hn(hx))
        hx = (1 - i) * n + i * hx
        return hx


class BayesianGRUCell(AbstractGRUCell):
    def __init__(self, input_size, hidden_size,
                 bias_ih=True, bias_hh=False,
                 dropout=0.25, af='tanh'):
        super(BayesianGRUCell, self).__init__(input_size, hidden_size,
                                              bias_ih, bias_hh)
        self.set_dropout(dropout)
        self.af = af

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.drop_ir = SequentialDropout(p=dropout)
        self.drop_ii = SequentialDropout(p=dropout)
        self.drop_in = SequentialDropout(p=dropout)
        self.drop_hr = SequentialDropout(p=dropout)
        self.drop_hi = SequentialDropout(p=dropout)
        self.drop_hn = SequentialDropout(p=dropout)

    def end_of_sequence(self):
        self.drop_ir.end_of_sequence()
        self.drop_ii.end_of_sequence()
        self.drop_in.end_of_sequence()
        self.drop_hr.end_of_sequence()
        self.drop_hi.end_of_sequence()
        self.drop_hn.end_of_sequence()

    def forward(self, x, hx=None):
        if hx is None:
            hx = Variable(x.data.new().resize_((x.size(0), self.hidden_size)).fill_(0))
        x_ir = self.drop_ir(x)
        x_ii = self.drop_ii(x)
        x_in = self.drop_in(x)
        x_hr = self.drop_hr(hx)
        x_hi = self.drop_hi(hx)
        x_hn = self.drop_hn(hx)
        r = F.sigmoid(self.weight_ir(x_ir) + self.weight_hr(x_hr))
        i = F.sigmoid(self.weight_ii(x_ii) + self.weight_hi(x_hi))
        n = getattr(F, self.af)(self.weight_in(x_in) + r * self.weight_hn(x_hn))
        hx = (1 - i) * n + i * hx
        return hx


class AbstractGRU(nn.Module):
    def __init__(self, input_size, hidden_size,
                 bias_ih=True, bias_hh=False):
        super(AbstractGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self._load_gru_cell()

    def _load_gru_cell(self):
        raise NotImplementedError

    def forward(self, x, hx=None, max_length=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        if max_length is None:
            max_length = seq_length
        output = []
        for i in range(max_length):
            hx = self.gru_cell(x[:, i, :], hx=hx)
            output.append(hx.view(batch_size, 1, self.hidden_size))
        output = torch.cat(output, 1)
        return output, hx


class BayesianGRU(AbstractGRU):
    def __init__(self, input_size, hidden_size,
                 bias_ih=True, bias_hh=False,
                 dropout=0.25, return_last=True, af='tanh'):
        self.dropout = dropout
        self.return_last = return_last
        self.af = af
        super(BayesianGRU, self).__init__(input_size, hidden_size,
                                          bias_ih, bias_hh)

    def _load_gru_cell(self):
        self.gru_cell = BayesianGRUCell(self.input_size, self.hidden_size,
                                        self.bias_ih, self.bias_hh,
                                        dropout=self.dropout, af=self.af)

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.gru_cell.set_dropout(dropout)

    def forward(self, x, lengths=None):
        hx = None
        batch_size = x.size(0)
        seq_length = x.size(1)
        max_length = seq_length
        output = []
        for i in range(max_length):
            hx = self.gru_cell(x[:, i, :], hx=hx)
            output.append(hx.view(batch_size, 1, self.hidden_size))
        self.gru_cell.end_of_sequence()
        output = torch.cat(output, 1)
        # TODO

        if self.return_last == 'all':
            self.all_hiddens = output.clone()
            x = output

            batch_size = x.size(0)

            mask = x.data.new().resize_as_(x.data).fill_(0)
            for i in range(batch_size):
                mask[i][lengths[i] - 1].fill_(1)
            mask = Variable(mask)
            x = x.mul(mask)
            x = x.sum(1).view(batch_size, 2400)
            return output, x
        if self.return_last == 'l':
            self.all_hiddens = output.clone()
            x = output

            batch_size = x.size(0)

            mask = x.data.new().resize_as_(x.data).fill_(0)
            for i in range(batch_size):
                mask[i][lengths[i] - 1].fill_(1)
            mask = Variable(mask)
            x = x.mul(mask)
            x = x.sum(1).view(batch_size, 2400)
            return output, x, lengths

        if self.return_last:
            self.all_hiddens = output.clone()
            x = output

            batch_size = x.size(0)

            mask = x.data.new().resize_as_(x.data).fill_(0)
            for i in range(batch_size):
                mask[i][lengths[i] - 1].fill_(1)
            mask = Variable(mask)
            x = x.mul(mask)
            x = x.sum(1).view(batch_size, 2400)
            return x

        else:
            return output


class RelationFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, R):
        super(RelationFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.R = R
        self.list_linear_hv = nn.ModuleList(
            [nn.Linear(input_dim1, hidden_dim) for _ in range(R)])
        self.list_linear_hq = nn.ModuleList(
            [nn.Linear(input_dim2, hidden_dim) for _ in range(R)])

    def forward(self, input_v, input_q):
        x_mm = []
        for i in range(self.R):
            x_hv = self.list_linear_hv[i](input_v)
            x_hq = self.list_linear_hq[i](input_q)
            x_mm.append(torch.bmm(x_hv, x_hq.transpose(1, 2)))
        x_mm = torch.stack(x_mm, dim=1)
        return x_mm

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)

class SEN(nn.Module):
    def __init__(self, input_dim, R): # 512, 4
        super(SEN, self).__init__()
        self.input_dim = input_dim
        assert input_dim % R ==0
        self.inner_dim = int(self.input_dim / R) # 9
        self.R = R
        self.conv1 = MyConv1d(self.input_dim, self.inner_dim, kernel_size=1, stride=1, seed=None, p=0.5, af='relu', dim=1)
        self.conv2 = MyConv1d(self.inner_dim, self.input_dim, kernel_size=1, stride=1, seed=None, p=0.5, af='sigmoid', dim=1)


    def forward(self, inputs1, inputs2):
        h = bmul(inputs1, inputs2) # bs x 36 x 512
        h_avg = torch.mean(h, dim=1)   # bs x 36
        h_avg = h_avg.unsqueeze(1) # bs x 1 x 36
        c1 = self.conv1(h_avg)   # bs x 1 x 9
        c2 = self.conv2(c1).squeeze() # bs x 36
        h_r = bmul(c2, h)

        return h_r
