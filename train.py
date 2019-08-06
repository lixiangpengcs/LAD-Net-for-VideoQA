import os
import time
import torch
import torch.nn as nn
import utils
import numpy as np
from torch.autograd import Variable


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1]
    pred_y = logits.data.cpu().numpy().squeeze()
    target_y = labels
    scores = sum(pred_y==target_y)
    return scores

def load_lossfunc(model):
    if model.model_name=='FrameQA':
        loss_func = torch.nn.CrossEntropyLoss()
    elif model.model_name=='Count':
        loss_func = torch.nn.MSELoss(reduction='elementwise_mean')
    elif model.model_name=='Trans' or model.model_name=='Action':
        loss_func = torch.nn.MultiMarginLoss()
    else:
        raise ValueError('Unknown task.')
    return loss_func

def margin_loss(pred, label):
    batch_agg_index = torch.from_numpy( np.concatenate(np.tile(np.arange(label.size(0)).reshape([label.size(0), 1]),
                                             [1, 5])) * 5) #[batch*mutil_choice]
    ans_agg_index = label.unsqueeze(1).repeat(1, 5).view(-1)
    index = Variable(batch_agg_index + ans_agg_index.data.cpu()).cuda().unsqueeze(1)
    gather = torch.gather(pred, 0, index)
    x = Variable(torch.zeros(pred.size(0),1)).cuda()
    y = 1.0 - gather + pred
    margin_los = torch.max(x, y)
    margin_los = torch.sum(margin_los)/pred.size(0)
    return margin_los


def train(model, train_loader, eval_loader, params, opt=None):
    lr_default = 1e-3 if eval_loader is not None else 7e-4
    lr_decay_step = 2
    lr_decay_rate = .5  # .25
    lr_decay_epochs = range(10, 30, lr_decay_step) if eval_loader is not None else range(10, 30, lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    saving_epoch = 3
    grad_clip = .25

    utils.create_dir(params['output']% (len(params['scale']), params['sub_nums'], params['reasonSteps'], str(params['lambda']),model.model_name))
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default) \
        if opt is None else opt
    logger = utils.Logger(os.path.join(params['output']% (len(params['scale']), params['sub_nums'], params['reasonSteps'], str(params['lambda']), model.model_name), 'log.txt'))

    # utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
                 (lr_default, lr_decay_step, lr_decay_rate, grad_clip))

    utils.create_dir(params['output'] % (len(params['scale']), params['sub_nums'], params['reasonSteps'], str(params['lambda']), model.model_name))
    # optim = torch.optim.Adamax(model.parameters())  #SGD(, lr=0.05, momentum=0.9)
    logger = utils.Logger(os.path.join(params['output'] % (len(params['scale']), params['sub_nums'], params['reasonSteps'], str(params['lambda']), model.model_name), 'log.txt'))
    if model.model_name=='Count':
        best_eval_score = 99999999
    else:
        best_eval_score = 0
    loss_func = load_lossfunc(model)

    for epoch in range(params['epochs']):
        total_loss = 0
        t = time.time()

        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            logger.write('lr: %.6f' % optim.param_groups[0]['lr'])

        print('===========Epoch %d \'s train=========='% (epoch))
        for i, (v, q, q_embed, a) in enumerate(train_loader):
            # reshape multi-choice question: batch_size x num_choide x max_length ==> batch_size*num_choide x max_length
            q = np.array(q)
            q = torch.from_numpy(q.reshape(-1,q.shape[-1]))
            q_embed = np.array(q_embed)
            q_embed = torch.from_numpy(q_embed.reshape(-1, q_embed.shape[-2], q_embed.shape[-1]))
            q_embed = Variable(q_embed.cuda())
            v = np.array(v)
            v = np.tile(v, [1, model.num_choice]).reshape(-1,v.shape[-2], v.shape[-1])
            v = Variable(torch.from_numpy(v).cuda())
            q = Variable(q.cuda())
            a = a.type(torch.LongTensor)
            if model.model_name=='Count':
                a = Variable(a.cuda()).float()
            else:
                a = Variable(a.cuda())

            pred, dis_loss = model(v, q, q_embed, a)
            loss1 = loss_func(pred, a)
            loss = loss1 + dis_loss * params['lambda']
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optim.step()
            optim.zero_grad()

            total_loss += loss.item() * v.size(0)
            if i % 100==0:
                print('iter: ', i, 'loss: %6f'%loss.item(), 'disagreement: %6f'%dis_loss.item())


        total_loss /= len(train_loader.dataset)
        logger.write('\ttrain_loss: %.2f' % (total_loss))
        model.train(False)
        print('===========Epoch %d \'s test==========' % (epoch))
        eval_score= model.evaluate(eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\teval score: %.2f ' % (100 * eval_score))
        if model.model_name=='Count':
            if eval_score < best_eval_score:
                model_path = os.path.join(params['output'] % (len(params['scale']), params['sub_nums'], params['reasonSteps'],
                                                              str(params['lambda']), model.model_name), 'model.pth')
                torch.save(model.state_dict(), model_path)
                best_eval_score = eval_score
            logger.write('\tcurrent best eval score: %.2f ' % (best_eval_score))
        else:
            if eval_score > best_eval_score:
                model_path = os.path.join(params['output'] % (len(params['scale']), params['sub_nums'], params['reasonSteps'],
                                                              str(params['lambda']), model.model_name), 'model.pth')
                torch.save(model.state_dict(), model_path)
                best_eval_score = eval_score
            logger.write('\tcurrent best eval score: %.2f ' % (100 * best_eval_score))
    logger.write('\tfinal best eval score: %.2f ' % (100 * best_eval_score))


